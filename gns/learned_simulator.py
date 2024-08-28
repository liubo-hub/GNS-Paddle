from typing import Dict
import numpy as np
import paddle
import tensorflow as tf
from gns_paddle import connectivity_utils
from graph_network import EncoderProcesserDecoder


class LearnedSimulator(paddle.nn.Layer):
    """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

    def __init__(
            self,
            particle_dimensions: int,
            nnode_in: int, nedge_in:int,
            latent_dim: int,
            nmessage_passing_steps: int,
            nmlp_layers: int,
            mlp_hidden_dim: int,
            connectivity_radius: float, boundaries: np.ndarray,
            normalization_stats: Dict,
            nparticle_types: int,
            particle_type_embedding_size,
            device='cpu'):
        """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      boundaries: Array of 2-tuples, containing the lower and upper boundaries
        of the cuboid containing the particles along each dimensions, matching
        the dimensionality of the problem.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      device: Runtime device (cuda or cpu).

    """
        super(LearnedSimulator, self).__init__()
        self._boundaries = boundaries
        self._connectivity_radius = connectivity_radius
        self._normalization_stats = normalization_stats
        self._nparticle_types = nparticle_types

        # Particle type embedding has shape (9, 16)
        self._particle_type_embedding = paddle.nn.Embedding(num_embeddings=
                                                            nparticle_types, embedding_dim=particle_type_embedding_size)
        # self._encode_process_decode = EncoderProcesserDecoder(
        #     nnode_in_features=nnode_in, nnode_out_features=
        #     particle_dimensions, nedge_in_features=nedge_in, latent_dim=
        #     latent_dim, nmessage_passing_steps=nmessage_passing_steps,
        #     nmlp_layers=nmlp_layers, mlp_hidden_dim=mlp_hidden_dim)
        self._encode_process_decode = EncoderProcesserDecoder(
            message_passing_num=nmessage_passing_steps,
            node_input_size=nnode_in,
            edge_input_size=nedge_in,
            hidden_size=latent_dim,
            nparticle_dimensions=particle_dimensions
        )
        self._device = device

    def forward(self):
        """Forward hook runs on class instantiation"""
        pass


    def _encoder_preprocessor(
            self,
            position_sequence: paddle.Tensor,
            nparticles_per_example: paddle.Tensor,
            particle_types: paddle.Tensor):
        """Extracts important features from the position sequence. Returns a tuple
    of node_features (nparticles, 30), edge_index (nparticles, nparticles), and
    edge_features (nparticles, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
    """
        nparticles = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1]
        velocity_sequence = time_diff(position_sequence)
        # Get connectivity of the graph with shape of (nparticles, 2)
        (senders, receivers, n_edge) = connectivity_utils._compute_connectivity_for_batch(
            most_recent_position, nparticles_per_example, self.
            _connectivity_radius)
        node_features = []
        senders = paddle.to_tensor(senders, dtype='int64')
        receivers = paddle.to_tensor(receivers, dtype='int64')
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self._normalization_stats['velocity']
        normalized_velocity_sequence = (velocity_sequence - velocity_stats[
            'mean']) / velocity_stats['std']
        flat_velocity_sequence = paddle.reshape(normalized_velocity_sequence, [nparticles, -1])

        # There are 5 previous steps, with dim 2
        # node_features shape (nparticles, 5 * 2 = 10)
        node_features.append(flat_velocity_sequence)
        # Normalized clipped distances to lower and upper boundaries.
        # boundaries are an array of shape [num_dimensions, 2], where the second
        # axis, provides the lower/upper boundaries.
        # boundaries = paddle.to_tensor(data=self._boundaries, stop_gradient=not False).astype(dtype='float32')
        boundaries = self._boundaries
        distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
        distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position
        distance_to_boundaries = paddle.concat(x=[distance_to_lower_boundary, distance_to_upper_boundary], axis=1)
        normalized_clipped_distance_to_boundaries = paddle.clip(x=distance_to_boundaries / self._connectivity_radius,min=-1.0, max=1.0)

        # The distance to 4 boundaries (top/bottom/left/right)
        # node_features shape (nparticles, 10+4)
        node_features.append(normalized_clipped_distance_to_boundaries)
        if self._nparticle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(
                particle_types)
            node_features.append(particle_type_embeddings)
        edge_features = []

        normalized_relative_displacements = (paddle.gather(most_recent_position, senders, axis=0) - paddle.gather(most_recent_position, receivers, axis=0))/self._connectivity_radius
        # normalized_relative_displacements = (most_recent_position[senders, :] - most_recent_position[receivers, :]) / self._connectivity_radius

        edge_features.append(normalized_relative_displacements)
        normalized_relative_distances = paddle.linalg.norm(x=normalized_relative_displacements, axis=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)

# -----------------------------------------------------------
#         edge_features.append(paddle.divide(paddle.to_tensor(1.0),
#                                            paddle.linalg.norm(normalized_relative_displacements, p=1, axis=-1,
#                                                               keepdim=True) + paddle.to_tensor(0.0001)))
#         edge_features.append(
#             paddle.divide(normalized_relative_displacements, normalized_relative_distances + paddle.to_tensor(0.0001)))
#         edge_features.append(
#             paddle.divide(paddle.to_tensor(1.0), normalized_relative_distances + paddle.to_tensor(0.0001)))
#         edge_features.append(paddle.divide(normalized_relative_displacements,
#                                            paddle.linalg.norm(normalized_relative_displacements, p=3, axis=-1,
#                                                               keepdim=True) + paddle.to_tensor(0.0001)))

        return paddle.concat(x=node_features, axis=-1), paddle.stack(x=[
            senders, receivers]), paddle.concat(x=edge_features, axis=-1)

    def _decoder_postprocessor(
            self,
            normalized_acceleration: paddle.Tensor,
            position_sequence: paddle.Tensor) -> paddle.Tensor:
        """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, dim).
      position_sequence: Position sequence of shape (nparticles, dim).

    Returns:
      torch.tensor: New position of the particles.

    """
        acceleration_stats = self._normalization_stats['acceleration']
        acceleration = normalized_acceleration * acceleration_stats['std'
        ] + acceleration_stats['mean']
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]
        new_velocity = most_recent_velocity + acceleration
        new_position = most_recent_position + new_velocity
        return new_position

    def predict_positions(
            self,
            current_positions: paddle.Tensor,
            nparticles_per_example: paddle.Tensor,
            particle_types: paddle.Tensor) -> paddle.Tensor:
        """Predict position based on acceleration.

    Args:
      current_positions: Current particle positions (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      next_positions (torch.tensor): Next position of particles.
    """
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types)
        predicted_normalized_acceleration = self._encode_process_decode(
            node_features, edge_index, edge_features)
        next_positions = self._decoder_postprocessor(
            predicted_normalized_acceleration, current_positions)
        return next_positions
    def predict_position(self,
                         next_position: paddle.Tensor,
                         position_sequence: paddle.Tensor,
                         ):
        nparticle = position_sequence.shape[0]
        present_velocity = time_diff(position_sequence)[:, -1]

        predicted_position =present_velocity + predicted_acceleration/2
        micro_scale = (tf.norm(predicted_position - next_position)) / nparticle
        macro_scale =tf.norm(tf.reduce_sum(predicted_position)/nparticle - tf.reduce_sum(next_position)/nparticle)
        position_loss = 0.6 * micro_scale + 0.4 * macro_scale

        return position_loss
    def predict_accelerations(
            self,
            next_positions: paddle.Tensor,
            position_sequence_noise: paddle.Tensor,
            position_sequence: paddle.Tensor,
            nparticles_per_example: paddle.Tensor,
            particle_types: paddle.Tensor):
        """Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized accelerations.

    """
        # Add noise to the input position sequence.
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Perform the forward pass with the noisy position sequence.
        node_features, edge_index, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types)

        global predicted_acceleration
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, edge_features)
        predicted_acceleration = predicted_normalized_acceleration
        # Calculate the target acceleration, using an `adjusted_next_position `that
        # is shifted by the noise in the last input position.
        next_position_adjusted = next_positions + position_sequence_noise[:, -1]

        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence)

        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_position: paddle.Tensor, position_sequence: paddle.Tensor):
        """Inverse of `_decoder_postprocessor`.

    Args:
      next_position: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.

    Returns:
      normalized_acceleration (torch.tensor): Normalized acceleration.

    """
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity
        acceleration_stats = self._normalization_stats['acceleration']
        normalized_acceleration = (acceleration - acceleration_stats['mean']
                                   ) / acceleration_stats['std']
        return normalized_acceleration

    def save(self, path: str = 'model.pt'):
        """Save model state

    Args:
      path: Model path
    """
        paddle.save(obj=self.state_dict(), path=path, protocol=4)

    def load(self, path: str):
        """Load model state from file

    Args:
      path: Model path
    """
        self.set_state_dict(state_dict=paddle.load(path=path))


def time_diff(position_sequence: paddle.Tensor) -> paddle.Tensor:
    """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
    return position_sequence[:, 1:] - position_sequence[:, :-1]
