import os, random

original_path = '/home/aiteam/tykim/idcard/data/img_align_celeba'

# os.listdir("/home/aiteam/tykim/idcard/data/img_align_celeba")

# filename = random.choice(os.listdir("/home/aiteam/tykim/idcard/data/img_align_celeba"))

# print(os.path.join(original_path, filename))
# filepath = os.path.join(original_path, filename)
# os.system(f'cp {filepath} /home/aiteam/tykim/temp')

filelist = os.listdir(original_path)
print(len(filelist))
filenames = random.sample(filelist, 1800)
for filename in filenames:
  filepath = os.path.join(original_path, filename)
  os.system(f'cp {filepath} /home/aiteam/tykim/temp/celeb_2K')