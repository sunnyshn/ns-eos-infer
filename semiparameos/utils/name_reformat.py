import os
import numpy as np
import h5py

def rename_eoss(eos_path, new_eos_path, old_mm_file, mm_file_num):
    # check that the new path file directory is there
    if not os.path.isdir(new_eos_path):
        os.makedirs(new_eos_path) 
    # doing this within the same directory will cause overwritting of the same file multiple times, which causes data to be lost... :(
    try: 
        print(f"Renaming EoS {old_mm_file} to eos_mmpoly{mm_file_num}_clean.out")
        os.rename(os.path.join(eos_path, old_mm_file), os.path.join(new_eos_path, f"eos_mmpoly{mm_file_num}_clean.out"))
    except:
        raise ("File cannot be replaced.")

if __name__ == "__main__":
    
    path = "/home/sunny.ng/semiparameos/set_exp_0.16/eos"
    new_path = "/home/sunny.ng/semiparameos/set_exp_0.16/eos_rf"
    eos_files = os.listdir(path)
    
    for num, file in enumerate(eos_files):
        rename_eoss(path, new_path, file, num)
    
