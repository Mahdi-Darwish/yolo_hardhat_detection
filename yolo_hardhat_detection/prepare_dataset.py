import xml.etree.ElementTree as ET
import os
import random
import shutil

CLASSES = ['head', 'helmet'] # head = 0, helmet = 1, person = 2
random.seed(42)  # so the split is reproducible every time you run it
RAW_IMAGES_DIR = 'dataset_raw/images'
RAW_ANNOTATIONS_DIR = 'dataset_raw/annotations'
OUTPUT_DIR = 'dataset'
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
# test ratio is the remainder (0.1)


def convert_xml_to_yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in CLASSES:
            continue  # skip anything unexpected
        class_id = CLASSES.index(class_name)
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        box_width = (xmax - xmin) / img_width
        box_height = (ymax - ymin) / img_height
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    return yolo_lines
def create_folders():
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{OUTPUT_DIR}/{split}/images', exist_ok=True)
        os.makedirs(f'{OUTPUT_DIR}/{split}/labels', exist_ok=True)
def process_split(xml_filenames, split_name):
    for xml_filename in xml_filenames:
        xml_path = os.path.join(RAW_ANNOTATIONS_DIR, xml_filename)

        # the image has the same name, just .png instead of .xml
        image_filename = xml_filename.replace('.xml', '.png')
        image_path = os.path.join(RAW_IMAGES_DIR, image_filename)

        if not os.path.exists(image_path):
            print(f"WARNING: missing image for {xml_filename}, skipping")
            continue

        yolo_lines = convert_xml_to_yolo(xml_path)

        # write the label .txt file
        label_filename = xml_filename.replace('.xml', '.txt')
        label_dest = os.path.join(OUTPUT_DIR, split_name, 'labels', label_filename)
        with open(label_dest, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # copy the image
        image_dest = os.path.join(OUTPUT_DIR, split_name, 'images', image_filename)
        shutil.copy(image_path, image_dest)

    print(f"Finished processing {split_name}: {len(xml_filenames)} files")
if __name__ == '__main__':
    xml_files = [f for f in os.listdir(RAW_ANNOTATIONS_DIR) if f.endswith('.xml')]
    random.shuffle(xml_files)

    total = len(xml_files)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = xml_files[:train_end]
    val_files = xml_files[train_end:val_end]
    test_files = xml_files[val_end:]

    print(f"Total: {total}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    create_folders()
    process_split(train_files, 'train')
    process_split(val_files, 'val')
    process_split(test_files, 'test')

    print("Done! Dataset is ready in the 'dataset' folder.")