from PIL import Image
import glob
import os

##
root_path = "/home/fujii/hdd/BF-C2DL-HSC/02/root"
lap_num = 11
train_num = 10
##

def main(root_path, lap_num, train_num):
    savepath = os.path.join(root_path, f"traindata_gif")
    os.makedirs(savepath, exist_ok=True)

    files_list = []
    for lap in range(lap_num):
        files = sorted(glob.glob(os.path.join(root_path, f"lap{lap}/traindata/check/*.png")))
        files_list.append(files)

    for tra in range(train_num):
        frames = []
        for lap in range(lap_num):
            new_frame = Image.open(files_list[lap][tra])
            frames.append(new_frame)
            pass
        save_path_vector = os.path.join(savepath, f"frame{tra:02}.gif")
        frames[0].save(save_path_vector, format="GIF", append_images=frames[1:], save_all=True, duration=400, loop=0)
    
    # images = list(map(lambda file: Image.open(file), files))

    # images[0].save('out.gif', save_all=True, append_images=images[1:], duration=400, loop=0)




if __name__ == "__main__":
    main(root_path, lap_num, train_num)
    print("finished")