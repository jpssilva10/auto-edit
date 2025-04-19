import os
from ultralytics import YOLO
import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance


# Function to list image files in a folder
def list_image_files(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Folders containing images
input_folder = os.path.join(os.getcwd(), "images\\input")
output_folder = os.path.join(os.getcwd(), "images\\output")

# List all image files in the folder
image_files = list_image_files(input_folder)

# Load the watermark image
watermark_path = os.path.join(os.getcwd(), "watermark.png")
watermark = Image.open(watermark_path).convert("RGBA")

# Read YOLO model
model = YOLO("yolov8m-seg.pt")

# if you want all classes
yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

conf = 0.4

# Loop through all image files
for image_file in image_files:
    # Construct the full path to the image file
    img_path = os.path.join(input_folder, image_file)
    img = cv2.imread(img_path)
    height, width, channels = img.shape


    # Segmentation
    results = model.predict(img, conf=conf)
    colors = [random.choices(range(256), k=3) for _ in classes_ids]
        # Check if results is empty
    if not results:
        print("No results found for this image")
    else:
        print(results)

        for result in results:
            if result.boxes and len(result.boxes.cls) > 0:  # Check if boxes and classes exist
                # Iterate through each class in result.boxes.cls
                for i, cls in enumerate(result.boxes.cls):
                    class_id = int(cls.item())  # Convert tensor to scalar using .item()
                    
                    if class_id == 2:
                        for mask, box in zip(result.masks.xy, result.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert to list of scalars

                            # Convert coordinates to integers
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            w = x2 - x1
                            h = y2 - y1
                            print(f"Box: ({x1}, {y1}, {x2}, {y2}), width: {w}, height: {h}")

                            # Ensure the cropped region maintains a 4:3 aspect ratio
                            aspect_ratio = 4 / 3
                            if w / h > aspect_ratio:
                                # Adjust height to match 4:3 aspect ratio
                                new_h = int(w / aspect_ratio)
                                h_diff = new_h - h
                                crop_y1 = max(0, y1 - h_diff // 2)
                                crop_y2 = min(height, y2 + h_diff // 2)
                                crop_x1, crop_x2 = x1, x2
                            else:
                                # Adjust width to match 4:3 aspect ratio
                                new_w = int(h * aspect_ratio)
                                w_diff = new_w - w
                                crop_x1 = max(0, x1 - w_diff // 2)
                                crop_x2 = min(width, x2 + w_diff // 2)
                                crop_y1, crop_y2 = y1, y2

                            # Add margin as a percentage of the original image size
                            margin_percentage = 0.1  # 10% margin
                            margin_w = int(width * margin_percentage)
                            margin_h = int(height * margin_percentage)

                            crop_x1 = max(0, crop_x1 - margin_w)
                            crop_y1 = max(0, crop_y1 - margin_h)
                            crop_x2 = min(width, crop_x2 + margin_w)
                            crop_y2 = min(height, crop_y2 + margin_h)

                            cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                            
                            # Convert to PIL for watermarking
                            pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)).convert("RGBA")

                            # Resize watermark to fit the cropped image
                            watermark_width, watermark_height = watermark.size
                            aspect_ratio = watermark_width / watermark_height
                            new_width = 200 #int(pil_img.width * 0.2)  # Adjust size as necessary
                            new_height = int(new_width / aspect_ratio)
                            watermark_resized = watermark.resize((new_width, new_height), Image.Resampling.LANCZOS)

                            # Position the watermark in the center of the cropped image
                            wm_x = (pil_img.width - new_width) // 2
                            wm_y = (pil_img.height - new_height) // 2

                            # Create an alpha mask
                            alpha_mask = watermark_resized.split()[3]
                            alpha_mask = ImageEnhance.Brightness(alpha_mask).enhance(0.1)  # Adjust transparency level (0.1 for 10%)

                            # Paste the watermark onto the image using the alpha mask
                            #pil_img.paste(watermark_resized, (wm_x, wm_y), alpha_mask)

                            # Convert back to OpenCV format
                            final_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
                            # Save the final image
                            # output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_cropped.jpg")
                            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_cropped{os.path.splitext(image_file)[1]}")
                            cv2.imwrite(output_path, final_img)
            else:
                print("No boxes found for this image")











        # Process each detection
        # for result in results:
        #     class_id = int(result.boxes.cls)
        #     if(class_id == 2):
        #         for mask, box in zip(result.masks.xy, result.boxes):
        #             # x1, y1, x2, y2, conf, class_id = result
        #             x1, y1, x2, y2 = box.xyxy[0]

        #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #             w = x2 - x1
        #             h = y2 - y1

        #             # Add margin
        #             # margin_w, margin_h = int(w * 0.1), int(h * 0.1)
        #             margin_w, margin_h = 200, 200
        #             crop_x1 = max(0, x1 - margin_w)
        #             crop_y1 = max(0, y1 - margin_h)
        #             crop_x2 = min(width, x2 + margin_w)
        #             crop_y2 = min(height, y2 + margin_h)
        #             cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
                    
        #             # Convert to PIL for watermarking
        #             pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)).convert("RGBA")

        #             # Resize watermark to fit the cropped image
        #             watermark_width, watermark_height = watermark.size
        #             aspect_ratio = watermark_width / watermark_height
        #             new_width = 512 #int(pil_img.width * 0.2)  # Adjust size as necessary
        #             new_height = int(new_width / aspect_ratio)
        #             watermark_resized = watermark.resize((new_width, new_height), Image.Resampling.LANCZOS)

        #             # Position the watermark in the center of the cropped image
        #             wm_x = (pil_img.width - new_width) // 2
        #             wm_y = (pil_img.height - new_height) // 2

        #             # Create an alpha mask
        #             alpha_mask = watermark_resized.split()[3]
        #             alpha_mask = ImageEnhance.Brightness(alpha_mask).enhance(0.1)  # Adjust transparency level (0.1 for 10%)

        #             # Paste the watermark onto the image using the alpha mask
        #             pil_img.paste(watermark_resized, (wm_x, wm_y), alpha_mask)

        #             # Convert back to OpenCV format
        #             final_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        #             # Save the final image
        #             output_path = os.path.join(image_folder, f"cropped_{os.path.splitext(image_file)[0]}.jpg")
        #             cv2.imwrite(output_path, final_img)

    # for result in results:
    #     for mask, box in zip(result.masks.xy, result.boxes):
    #         points = np.int32([mask])
    #         # cv2.polylines(img, points, True, (255, 0, 0), 1)
    #         color_number = classes_ids.index(int(box.cls[0]))
    #         # cv2.fillPoly(img, points, colors[color_number])
    #         b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
    #         left, top, right, bottom = box.xyxy[0]
    #         left = int(left)
    #         top = int(top)
    #         w = int(right) - left
    #         h = int(bottom) - top
    #         margin_w, margin_h = int(w * 0.1), int(h * 0.1)
    #         x = max(0, left - margin_w)
    #         y = max(0, top - margin_h)
    #         w = min(width - left, w + 2 * margin_w)
    #         h = min(height - top, h + 2 * margin_h)
    #         cropped_img = img[y:y+h, x:x+w]


            # Save the final image
            # output_path = os.path.join(image_folder, f"cropped_{os.path.splitext(image_file)[0]}.jpg")
            # cv2.imwrite(output_path, cropped_img)
# cv2.imshow("Image", img)
# cv2.waitKey(0)

