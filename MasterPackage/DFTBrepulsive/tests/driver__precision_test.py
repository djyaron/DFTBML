from deprecated.driver_v0 import *

if __name__ == '__main__':
    """
    Precision error occurs under certain cases.
    model.get_targets_from_dataset() was found to give different outputs
    when in_memory was set to True/False. The difference of the outputs
    reached (+/-) 2^-36. This can be reproduced by:
        1. 'target' = 'fm'. Save 'target' to a dictionary
        2. 'baseline' = 'pf' - 'pr'. Save 'baseline' to a dictionary
        3. 'target' - 'baseline' is now different from directly computing
           'fm' - 'pf' + 'pr'
    However, the issue did not occur when
        1. 'target' = 'fm' + 'pr' and 'baseline' = 'pf'
        2. 'target' = 'fm' - 'pf' and 'baseline' = 'pr'
           and compare 'target' + 'baseline' with 'fm' - 'pf' + 'pr'
    """

    # This is the original code
    src_path = '/home/francishe/Documents/DFTBrepulsive/Datasets/aed_1K.h5'
    with File(src_path, 'r') as src:
        dset = {mol: {'atomic_numbers': moldata['atomic_numbers'][()],
                      'coordinates': moldata['coordinates'][()],
                      'target': moldata[ALIAS2TARGET['fm']][()],
                      'baseline': moldata[ALIAS2TARGET['pf']][()] - moldata[ALIAS2TARGET['pr']][()]}
                for mol, moldata in src.items()}

    # Tests
    t0 = {}
    with File(src_path, 'r') as dataset:
        dtypes = [ALIAS2TARGET['fm'], ALIAS2TARGET['pf'], ALIAS2TARGET['pr']]
        fd = Fold.from_dataset(dataset)
        for mol, conf_arr in fd.items():
            t0[mol] = dataset[mol][dtypes[0]][conf_arr] - \
                      dataset[mol][dtypes[1]][conf_arr] + \
                      dataset[mol][dtypes[2]][conf_arr]

    t1 = get_targets_from_dataset(['fm', 'pf', 'pr'], fd, src_path)

    t2 = {}
    with File(src_path, 'r') as src:
        for mol, moldata in src.items():
            t2[mol] = moldata[ALIAS2TARGET['fm']][()] - \
                      moldata[ALIAS2TARGET['pf']][()] + \
                      moldata[ALIAS2TARGET['pr']][()]

    t3 = {}
    with File(src_path, 'r') as src:
        tmp = {}
        for mol, moldata in src.items():
            tmp[mol] = {}
            tmp[mol]['entry1'] = moldata[ALIAS2TARGET['fm']][()]
            tmp[mol]['entry2'] = moldata[ALIAS2TARGET['pf']][()] - \
                                 moldata[ALIAS2TARGET['pr']][()]
            t3[mol] = tmp[mol]['entry1'] - tmp[mol]['entry2']

    t4 = {}
    with File(src_path, 'r') as src:
        tmp = {}
        for mol, moldata in src.items():
            tmp[mol] = {}
            tmp[mol]['entry1'] = moldata[ALIAS2TARGET['fm']][()] - \
                                 moldata[ALIAS2TARGET['pf']][()]
            tmp[mol]['entry2'] = moldata[ALIAS2TARGET['pr']][()]
            t4[mol] = tmp[mol]['entry1'] + tmp[mol]['entry2']

    t5 = {}
    with File(src_path, 'r') as src:
        tmp = {}
        for mol, moldata in src.items():
            tmp[mol] = {}
            tmp[mol]['entry1'] = moldata[ALIAS2TARGET['fm']][()] + \
                                 moldata[ALIAS2TARGET['pr']][()]
            tmp[mol]['entry2'] = moldata[ALIAS2TARGET['pf']][()]
            t5[mol] = tmp[mol]['entry1'] - tmp[mol]['entry2']

    t6 = get_targets_from_dataset(data=dset, in_memory=True)

    def print_diff(a, b):
        tdiff = []
        for tsrc, tdes in zip(a.values(), b.values()):
            tdiff.append(tsrc - tdes)
        tdiff = np.concatenate(tdiff)
        ndiff = (tdiff != 0).sum()
        print(f"#diffs: {ndiff}. First 5: {np.where(tdiff != 0)[0][:5]}")

    print_diff(t0, t1)  # identical
    print_diff(t0, t2)  # identical
    print_diff(t0, t3)  # 218 different values
    print_diff(t0, t4)  # identical
    print_diff(t0, t5)  # identical
    print_diff(t0, t6)  # same as t0 vs t3