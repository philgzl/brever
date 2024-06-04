from brever.mixture import RandomMixtureMaker


def test_random_mixture_maker():
    rmm = RandomMixtureMaker(
        seed=0,
        speakers={'libri_.*', 'vctk_.*'},
        noises={'dcase_.*', 'demand_.*'},
        rooms={'surrey_.*', 'ash_.*'},
        decay=True,
        diffuse=True,
        noise_num=(0, 3),
    )
    for i in range(10):
        rmm()
