import os, shutil

target = './data/tiny-imagenet-200/val'
file = open(os.path.join(target, 'val_annotations.txt'), 'r')
lines = file.readlines()
source = os.path.join(target, 'images')
dest = os.path.join(target, 'labeled_val')
if not os.path.exists(dest):
    os.mkdir(dest)
for line in lines:
    info = line.split('\t')[:2]
    fn = info[0]
    cid = info[1]

    class_folder = os.path.join(dest, cid)
    if not os.path.exists(class_folder):
        os.mkdir(class_folder)

    img_path = os.path.join(source, fn)
    dest_path = os.path.join(dest, cid, fn)
    shutil.copy(img_path, dest_path)
    print("old_path:" + str(img_path))
    print("new_path:" + str(dest_path))