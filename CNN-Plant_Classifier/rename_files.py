import os

def rename_files(filename):
	count = 1
	for picname in os.listdir(filename):
		print(picname)
		if (picname == ".DS_Store"):
			pass
		else:
			if picname.endswith(".JPG"):
				extension = ".JPG"
			elif picname.endswith(".jpg"):
				extension = ".jpg"

			pic_rename = "img_"+str(count)+str(extension)
			dst = filename +"/"+pic_rename
			src = filename+"/"+picname

			os.rename(src,dst)
			count+=1

def file_check(filename):
	for pic_dir in os.listdir(filename):
		change_dir_names = filename+"/"+pic_dir
		if pic_dir == ".DS_Store":
			pass
		else:
			print(change_dir_names)
			rename_files(change_dir_names)


# if __name__ == "__main__":
# 	filename = "/Users/krishnavenkatramani/Desktop/Plant_AI/PlantVillage-Dataset/raw/color"
# 	for pic_dir in os.listdir(filename):
# 		change_dir_names = filename+"/"+pic_dir
# 		if pic_dir == ".DS_Store":
# 			pass
# 		else:
# 			print(change_dir_names)
# 			rename_files(change_dir_names)


