from brever.mixture import RandomMixtureMaker


def test_random_mixture_maker():
    rmm = RandomMixtureMaker(
        seed=0,
        speakers={'timit_.*', 'libri_.*'},
        noises={'dcase_.*', 'icra_.*'},
        rooms={'surrey_.*', 'ash_.*'},
    )
    for i in range(10):
        rmm.make()
