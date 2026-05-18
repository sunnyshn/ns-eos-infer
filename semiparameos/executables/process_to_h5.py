import h5py
import numpy as np
import pandas as pd
import os

def mm_eos_directory_to_h5(
        directory_path, indices_to_use, h5_outpath, 
        eos_per_mm=15,
        mm_template = lambda index: f"EoSNewRestricted-{index}"):
    h5file = h5py.File(h5_outpath, "w")
    # ns = h5file.create_group("ns")
    eos_group = h5file.create_group("eos")
    mm_ids  = []
    counter = 0
    for index in indices_to_use:
        for sample_id in range(eos_per_mm):
            try:
                print(os.path.join(directory_path, mm_template(index),
                                               f"eos-draw-{sample_id:04d}.csv"))
                try:
                    eos = pd.read_csv(os.path.join(directory_path, mm_template(index),
                                                    f"eos-draw-{sample_id:04d}.csv"))
                except:
                    print("EOS read in failure.")

                # try:
                #     macro = pd.read_csv(os.path.join(directory_path, mm_template(index),
                #                                 f"macro-eos-draw-{sample_id:04d}.csv"))
                # except:
                #     print("Macro read in failure.")
                print(type(eos.to_records()))

                # ns[f"eos_{counter:06d}"] = macro.to_records()
                eos_group[f"eos_{counter:06d}"] =  eos.to_records()
                mm_ids.append(index)
                counter += 1
            except FileNotFoundError:
                print("metamodel id", index, "extension #", sample_id, "not found" )
    eos_id = h5file.create_dataset("id", data=np.arange(counter))
    mm_id = h5file.create_dataset("mm_id", data=np.array(mm_ids))
    h5file.close()

if __name__ == "__main__":
    
    eos_folder = "marginalized_hyp"
    eos_path = f"/home/sunny.ng/semiparameos/result/{eos_folder}/"
    num_eos = int(len(os.listdir(eos_path))) 
    result_path = "/home/sunny.ng/semiparameos/generated_eoss"
    output_file = "marginalized_hyp_MMGP.h5"
    mm_eos_directory_to_h5(eos_path, indices_to_use=[n for n in range(num_eos)], h5_outpath=f"{result_path}/{output_file}")

