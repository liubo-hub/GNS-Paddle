from PIL import Image, ImageDraw, ImageFont
from absl import flags
import os
def Draw_Steps_Loss(Step,numpy_Loss):
    # create image
    width, height = 2000, 1600
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # Define The range of X and Y axes
    x_min, x_max = min(Step), max(Step)
    y_min, y_max = min(numpy_Loss), max(numpy_Loss)

    # Scaling factor， mapping data to image size
    x_scale = (width - 100) / (x_max - x_min)
    y_scale = (height - 100) / (y_max - y_min)

    # create The X and Y axes
    draw.line([(50, height - 50), (width - 50, height - 50)], fill=(0, 0, 0), width=2)  # X轴
    draw.line([(50, height - 50), (50, 50)], fill=(0, 0, 0), width=2)  # Y轴

    # Draw X-axis scale values
    x_tick_font = ImageFont.load_default()
    last_step = Step[-1]
    x = 50 + (last_step - x_min) * x_scale
    y = height - 50
    coordinate_text = f"{last_step}"
    text_width, text_height = draw.textsize(coordinate_text, font=x_tick_font)
    draw.text((x - text_width / 2, y + 5), coordinate_text, fill=(0, 0, 0), font=x_tick_font)


    # y_tick_font = ImageFont.load_default()
    # for i in range(20):
    #     y = height - 50 - i * ((height - 100) / 10)
    #     coordinate_text = f"{round(y_min + i * ((y_max - y_min) / 10), 2)}"
    #     text_width, text_height = draw.textsize(coordinate_text, font=y_tick_font)
    #     draw.text((25 - text_width, y - text_height / 2), coordinate_text, fill=(0, 0, 0), font=y_tick_font)

    # Draw a line chart
    for i in range(len(Step) - 1):
        x1 = 50 + (Step[i] - x_min) * x_scale
        y1 = height - 50 - (numpy_Loss[i] - y_min) * y_scale
        x2 = 50 + (Step[i + 1] - x_min) * x_scale
        y2 = height - 50 - (numpy_Loss[i + 1] - y_min) * y_scale
        draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)


        # font = ImageFont.load_default()
        # for i in range(len(Step)):
        #     x = 50 + (Step[i] - x_min) * x_scale
        #     y = height - 50 - (numpy_Loss[i] - y_min) * y_scale
        #     draw.text((x, y), f"({Step[i]}, {numpy_Loss[i]})", fill=(0, 0, 0), font=font)

    # save image
    data_path = flags.FLAGS.model_path
    file_path = os.path.join(data_path, "Step-Loss.png")
    image.save(file_path, quality=100)
    # show image
    image.show()
