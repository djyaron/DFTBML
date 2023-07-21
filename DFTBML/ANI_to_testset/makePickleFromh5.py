"""
Turns the cleaned h5 file dataset into a pickle file such as the one used in DFTBML.
"""
# %%
from h5_converter import h5_handler
import pickle

# %%

if __name__ == "__main__":
    k = h5_handler.h5_to_dict('cleaned_dsets/new_ANI1ccx_clean.h5')
    p_file = []
    for name in k.keys():
        #p_entry = {}
        #p_entry['name'] = name
        iconfig = 0
        for coordinates in k[name]['coordinates']:
            p_entry = {}
            p_entry['name'] = name
            p_entry['iconfig'] = k[name]['iconfig'][iconfig] # ! references original ani iconfig for regular position in clean dataset put just iconfig
            p_entry['atomic_numbers'] = k[name]['atomic_numbers']
            p_entry['coordinates'] = coordinates
            p_entry['targets'] = {}
            p_entry['targets']['dipole'] = k[name]['wb97x_dz.dipole'][iconfig]
            p_entry['targets']['charges'] = k[name]['wb97x_dz.cm5_charges'][iconfig]
            p_entry['targets']['Etot'] = k[name]['ccsd(t)_cbs.energy'][iconfig]

            iconfig += 1
            
            p_file.append(p_entry)

    new = 'cleaned_dsets/newclean_ANI.p'

    with (open(new, "wb")) as writefile:
        pickle.dump(p_file, writefile)

# %%
