import os, cv2, time

def main(dir1, dir2, dir3, sav_dir):
	# create a list of img file names for a given dir
	imgs = [file for file in os.listdir(dir1) if file.endswith('.png')]

	# sort this list
	imgs.sort()
	ct = 0
	
	for i in imgs:
		path1 = os.path.join(dir1, i)
		path2 = os.path.join(dir2, i)
		path3 = os.path.join(dir3, i)
		sav_path = os.path.join(sav_dir, i)
		if os.path.exists(path1) and os.path.exists(path2) and os.path.exists(path3):
			ct += 1
			img1 = cv2.imread(path1)
			img2 = cv2.imread(path2)
			img3 = cv2.imread(path3)
			imgh = cv2.hconcat([img1,img2, img3])
			cv2.imwrite(sav_path, imgh)
		
	return ct


if __name__ == "__main__":
	dir1 = input('Enter the full path of image dir (Left): ')
	dir2 = input('Enter the full path of image dir (Middle): ')
	dir3 = input('Enter the full path of image dir (Right): ')
	sav_dir = input('Enter the full path of output dir name: ')
	start_time = time.perf_counter()
	if not os.path.exists(sav_dir):
		os.makedirs(sav_dir)
	
	ct = main(dir1, dir2, dir3, sav_dir)
	print()
	print(ct, 'sets of image were processed and saved to :\n', sav_dir)
	print('Job Done (in seconds):', time.perf_counter() - start_time)

