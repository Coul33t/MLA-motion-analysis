import os
import shutil

#Be VERY careful with what you're doing here.

def main(motion_name="cut", property_name="lin_mean"):
    folder = r'C:\Users\quentin\Documents\Programmation\C++\MLA\Data\Speed'

    for d in os.listdir(folder):
        if motion_name in d:
            for sd in os.listdir(folder + '\\' + d):
                if property_name in sd:
                    print('Erasing {}'.format(folder + '\\' + d + '\\' +  sd))
                    shutil.rmtree(folder + '\\' + d + '\\' +  sd)



if __name__ == '__main__':
    main(motion_name="throw")