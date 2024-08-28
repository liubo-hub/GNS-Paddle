import collections
import glob
import json
import math
import os
import pickle
import re
import sys

import numpy as np
import paddle
from absl import app
from absl import flags
from tensorboardX import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gns_paddle import learned_simulator
import noise_utils
import reading_utils
import data_loader

flags.DEFINE_enum('mode', 'train', ['train', 'valid', 'rollout'], help='Train model, validation or rollout evaluation.')
flags.DEFINE_integer('batch_size', 1, help='The batch size.')
flags.DEFINE_float('noise_std', 0.00067, help='The std deviation of the noise.')
flags.DEFINE_string('data_path', 'datasets_WaterDropSample/', help='The dataset directory.')
flags.DEFINE_string('model_path', 'models/', help='The path for saving checkpoints of the model.')
flags.DEFINE_string('output_path', 'rollouts/', help='The path for saving outputs (e.g. rollouts).')
flags.DEFINE_string('model_file', None, help='Model filename (.pt) to resume from. Can also use "latest" to default to newest file.')
flags.DEFINE_string('train_state_file', 'train_state.pt', help='Train state filename (.pt) to resume from. Can also use "latest" to default to newest file.')
flags.DEFINE_integer('ntraining_steps', int(20000000.0), help='Number of training steps.')
flags.DEFINE_integer('nsave_steps', int(20000), help='Number of steps at which to save the model.')
flags.DEFINE_float('lr_init', 0.0001, help='Initial learning rate.')
flags.DEFINE_float('lr_decay', 0.1, help='Learning rate decay.')
flags.DEFINE_integer('lr_decay_steps', int(5000000.0), help='Learning rate decay steps.')
flags.DEFINE_integer('cuda_device_number', None,
                     help='CUDA device (zero indexed), default is None so default CUDA device will be used.')
Stats = collections.namedtuple('Stats', ['mean', 'std'])
writer = SummaryWriter(log_dir='../loss')
INPUT_SEQUENCE_LENGTH = 6
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

# Step = []
# Loss = []

def rollout(
        simulator: learned_simulator.LearnedSimulator,
        position: paddle.Tensor,
        particle_types: paddle.Tensor,
        n_particles_per_example: paddle.Tensor,
        nsteps: int,
        device):
    """Rolls out a trajectory by applying the model in sequence.

  Args:
    simulator: Learned simulator.
    features: Torch tensor features.
    nsteps: Number of steps.
  """
    initial_positions = position[:, :INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = position[:, INPUT_SEQUENCE_LENGTH:]
    current_positions = initial_positions
    predictions = []

    for step in range(nsteps):
        next_position = simulator.predict_positions(current_positions,
            nparticles_per_example=[n_particles_per_example],
            particle_types=particle_types
        )

        # Update kinematic particles from prescribed trajectory.
        kinematic_mask = (particle_types == KINEMATIC_PARTICLE_ID).clone(
            ).detach()
        next_position_ground_truth = ground_truth_positions[:, step]
        kinematic_mask = kinematic_mask.astype(dtype='bool')[:, None].expand(
            shape=[-1, current_positions.shape[-1]])
        next_position = paddle.where(condition=kinematic_mask, x=
            next_position_ground_truth, y=next_position)
        predictions.append(next_position)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end.
        current_positions = paddle.concat(x=[current_positions[:, 1:],
            next_position[:, None, :]], axis=1)

    predictions = paddle.stack(x=predictions)
    ground_truth_positions = ground_truth_positions.transpose(perm=[1, 0, 2])
    #
    loss = (predictions - ground_truth_positions) ** 2

    output_dict = {
        'initial_positions': initial_positions.transpose(perm=[1, 0, 2]).cpu().numpy(),
        'predicted_rollout': predictions.cpu().numpy(),
        'ground_truth_rollout': ground_truth_positions.cpu().numpy(),
        'particle_types': particle_types.cpu().numpy()}

    return output_dict, loss

def predict(device: str, FLAGS, flags, world_size):
    """Predict rollouts.

  Args:
    simulator: Trained simulator if not will undergo training.

  """
    metadata = reading_utils.read_metadata(FLAGS.data_path)
    simulator = _get_simulator(metadata, FLAGS.noise_std, FLAGS.noise_std,
        device)
    # Load simulator
    if os.path.exists(FLAGS.model_path + FLAGS.model_file):
        simulator.load(FLAGS.model_path + FLAGS.model_file)
    else:
        train(simulator, flags, world_size)
    simulator.to(device)
    simulator.eval()

    # Output path
    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)
    # Use `valid`` set for eval mode if not use `test`
    split = 'test' if FLAGS.mode == 'rollout' else 'valid'

    ds = data_loader.get_data_loader_by_trajectories(path=
        f'{FLAGS.data_path}{split}.npz')

    eval_loss = []
    with paddle.no_grad():
        for example_i, (positions, particle_type, n_particles_per_example) in enumerate(ds):
          if example_i==99:
            positions = paddle.to_tensor(positions, place=device)

            particle_type = paddle.to_tensor(particle_type, place=device)
            n_particles_per_example = paddle.to_tensor(data=[int(
                n_particles_per_example)], dtype='int32', place=device)

            nsteps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
            # Predict example rollout
            example_rollout, loss = rollout(simulator, positions,
                particle_type, n_particles_per_example, nsteps, device)

            example_rollout['metadata'] = metadata
            print('Predicting example {} loss: {}'.format(example_i, loss.mean()))
            eval_loss.append(paddle.flatten(x=loss))

            # Save rollout in testing
            if FLAGS.mode == 'rollout':
                example_rollout['metadata'] = metadata
                filename = f'rollout_{example_i}.pkl'
                filename = os.path.join(FLAGS.output_path, filename)
                with open(filename, 'wb') as f:
                    pickle.dump(example_rollout, f)
    print('Mean loss on rollout prediction: {}'.format(paddle.mean(x=paddle.concat(x=eval_loss))))



def train(device, flags, world_size):
    """Train the model.

  Args:
    rank: local rank
    world_size: total number of ranks
  """
    # if type(rank) == int:
    #     distribute.setup(rank, world_size)
    #     device = paddle.CUDAPlace(rank)  # 设置为 CUDAPlace，并传入 rank
    # else:
    #     device = paddle.CPUPlace()  # 设置为 CPUPlace
    metadata = reading_utils.read_metadata(flags['data_path'])
    serial_simulator = _get_simulator(metadata, flags['noise_std'], flags['noise_std'], device)
    simulator = serial_simulator.to(device)
    optimizer = paddle.optimizer.Adam(parameters=simulator.parameters(),
                                      learning_rate=flags['lr_init'] * world_size, weight_decay=0.0)
    step = 0

    # If model_path does exist and model_file and train_state_file exist continue training.
    if flags['model_file'] is not None:

        # find the latest model, assumes model and train_state files are in step.
        if flags['model_file'] == 'latest' and flags['train_state_file'] == 'latest':
            fnames = glob.glob(f"{flags['model_path']}*model*pt")
            max_model_number = 0
            expr = re.compile('.*model-(\\d+).pt')
            for fname in fnames:
                model_num = int(expr.search(fname).groups()[0])
                if model_num > max_model_number:
                    max_model_number = model_num
            flags['model_file'] = f'model-{max_model_number}.pt'
            flags['train_state_file'] = f'train_state-{max_model_number}.pt'

        if os.path.exists(flags['model_path'] + flags['model_file']
            ) and os.path.exists(flags['model_path'] + flags[
            'train_state_file']):

            # load model
            simulator.load(flags['model_path'] + flags['model_file'])

            # load train state
            train_state = paddle.load(path=flags['model_path'] + flags[
                'train_state_file'])
            # set optimizer state
            optimizer = paddle.optimizer.Adam(parameters=simulator.parameters(), weight_decay=0.0)
            optimizer.set_state_dict(state_dict=train_state['optimizer_state'])

            # set global train state
            step = train_state['global_train_state'].pop('step')
        else:
            msg = (
                f"Specified model_file {flags['model_path'] + flags['model_file']} and train_state_file {flags['model_path'] + flags['train_state_file']} not found."
                )
            raise FileNotFoundError(msg)

    simulator.train()
    simulator.to(device)

    dl = data_loader.get_data_loader_by_samples(
                            path=f"{flags['data_path']}train.npz",
                            input_length_sequence=INPUT_SEQUENCE_LENGTH,
                            batch_size=flags['batch_size'])
    not_reached_nsteps = True

    best_loss = 1.0
    try:
        while not_reached_nsteps:
            for (position, particle_type, n_particles_per_example), labels in dl:
                position = paddle.to_tensor(position, dtype='float32', place=device)
                particle_type = paddle.to_tensor(particle_type, dtype='int32', place=device)
                n_particles_per_example = paddle.to_tensor(n_particles_per_example, dtype='int32', place=device)
                labels = paddle.to_tensor(labels, dtype='float32', place=device)

                sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                    position,
                    noise_std_last_step=flags['noise_std']
                )
                sampled_noise = paddle.to_tensor(sampled_noise, dtype='float32', place=device)

                non_kinematic_mask = (particle_type != KINEMATIC_PARTICLE_ID)
                non_kinematic_mask = paddle.to_tensor(non_kinematic_mask, dtype='bool', place=device)
                # PaddlePaddle 中，应该使用 .reshape() 方法来更改张量的形状，而不是使用 .view()
                sampled_noise *= non_kinematic_mask.reshape([-1, 1, 1])

                pred_acc, target_acc = simulator.predict_accelerations(
                    next_positions=paddle.to_tensor(labels),
                    position_sequence_noise=paddle.to_tensor(sampled_noise),
                    position_sequence=paddle.to_tensor(position),
                    nparticles_per_example=paddle.to_tensor(n_particles_per_example),
                    particle_types=paddle.to_tensor(particle_type))

                # Calculate the loss and mask out loss on kinematic particles
                loss_normal = (pred_acc - target_acc) ** 2
                loss_normal = loss_normal.sum(axis=-1)
                # loss = loss_normal
                # ----------------
                pred_acc = pred_acc.sum(axis=-1)
                target_acc = target_acc.sum(axis=-1)
                loss_momentum = ((pred_acc - target_acc) ** 2) * (math.exp(-2))
                loss = (loss_normal + loss_momentum) / 2

                num_non_kinematic = non_kinematic_mask.sum()
                loss = paddle.where(condition=non_kinematic_mask.astype(
                    dtype='bool'), x=loss, y=paddle.zeros_like(x=loss))
                loss = loss.sum() / num_non_kinematic

                loss_position = simulator.predict_position(next_position=paddle.to_tensor(labels),
                                                           position_sequence=paddle.to_tensor(position))
                loss = loss + paddle.to_tensor(loss_position.numpy())

                loss.backward()
                optimizer.step()
                optimizer.clear_grad()

                if step % flags['nsave_steps'] == 0:
                    simulator.save(flags['model_path'] + 'model-' + str(step) + '.pt')
                    train_state = dict(optimizer_state=optimizer.state_dict(),
                                       global_train_state={'step': step})
                    paddle.save(obj=train_state, path=f"{flags['model_path']}train_state-{step}.pt", protocol=4)

                numpy_loss = float(loss)
                if numpy_loss < best_loss:
                    best_loss = numpy_loss
                    simulator.save(flags['model_path'] + 'model_min_loss' + '.pt')
                    train_state = dict(optimizer_state=optimizer.state_dict(),
                                       global_train_state={'step': step})
                    paddle.save(obj=train_state, path=f"{flags['model_path']}train_state_min_loss.pt", protocol=4)

                # 绘制Loss-step折线图
                # Step.append(step)
                # Loss.append(numpy_loss)
                if step % 20==0:
                    print(f'Training step: {step}/{flags["ntraining_steps"]}. Loss: {numpy_loss}.')
                writer.add_scalar('Loss-Step', numpy_loss, step)
                # Complete training
                if step >= flags['ntraining_steps']:
                    not_reached_nsteps = False
                    break
                step += 1

        # #输出Loss最小
        print(f"Loss最小为：{best_loss}")

    except KeyboardInterrupt:
        pass

def _get_simulator(
        metadata: json,
        acc_noise_std: float,
        vel_noise_std:float,
        device: str) ->learned_simulator.LearnedSimulator:
    """Instantiates the simulator.

  Args:
    metadata: JSON object with metadata.
    acc_noise_std: Acceleration noise std deviation.
    vel_noise_std: Velocity noise std deviation.
    device: PyTorch device 'cpu' or 'cuda'.
  """
    # Normalization stats
    normalization_stats = {
        'acceleration': {
            'mean': paddle.to_tensor(data=metadata['acc_mean'], dtype='float32'),
            'std': paddle.sqrt(x=paddle.to_tensor(data=metadata['acc_std'], dtype='float32') **2 +
                                 acc_noise_std ** 2)
        },
        'velocity': {
            'mean': paddle.to_tensor(data=metadata['vel_mean'], dtype='float32'),
            'std': paddle.sqrt(x=paddle.to_tensor(data=metadata['vel_std'],dtype='float32') ** 2 + vel_noise_std ** 2)
        }
    }

    simulator = learned_simulator.LearnedSimulator(
        particle_dimensions=metadata['dim'],
        nnode_in=37 if metadata['dim'] == 3 else 30,
        # nedge_in=metadata['dim'] + 9 if metadata['dim'] == 3 else metadata['dim'] + 7,
        nedge_in=metadata['dim'] + 1,
        latent_dim=128,
        nmessage_passing_steps=10,
        nmlp_layers=2,
        mlp_hidden_dim=128,
        connectivity_radius=metadata['default_connectivity_radius'],
        boundaries=np.array(metadata['bounds']),
        normalization_stats=normalization_stats,
        nparticle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
        device=device)

    return simulator


def main(_):
    """Train or evaluates the model.

  """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    FLAGS = flags.FLAGS
    myflags = {}
    myflags['data_path'] = FLAGS.data_path
    myflags['noise_std'] = FLAGS.noise_std
    myflags['lr_init'] = FLAGS.lr_init
    myflags['lr_decay'] = FLAGS.lr_decay
    myflags['lr_decay_steps'] = FLAGS.lr_decay_steps
    myflags['batch_size'] = FLAGS.batch_size
    myflags['ntraining_steps'] = FLAGS.ntraining_steps
    myflags['nsave_steps'] = FLAGS.nsave_steps
    myflags['model_file'] = FLAGS.model_file
    myflags['model_path'] = FLAGS.model_path
    myflags['train_state_file'] = FLAGS.train_state_file
    if FLAGS.mode == 'train':
        device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu').replace('cuda', 'gpu')
        if not os.path.exists(FLAGS.model_path):
            os.makedirs(FLAGS.model_path)
        world_size = paddle.device.cuda.device_count()
        print(f'world_size = {world_size}')
        train(device, myflags, world_size)

    elif FLAGS.mode in ['valid', 'rollout']:
        device = str('cuda' if paddle.device.cuda.device_count() >= 1 else 'cpu').replace('cuda', 'gpu')
        world_size = paddle.device.cuda.device_count()
        if (FLAGS.cuda_device_number is not None and paddle.device.cuda.
            device_count() >= 1):
            device = str(f'cuda:{int(FLAGS.cuda_device_number)}').replace(
                'cuda', 'gpu')
        predict(device, FLAGS, flags=myflags, world_size=world_size)


if __name__ == '__main__':
    app.run(main)
