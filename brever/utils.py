import shutil

import numpy as np

eps = np.finfo(float).eps


def pad(x, n, axis=0, where='right'):
    """Zero-padding along given axis.

    Parameters
    ----------
    x : array_like
        Input array.
    n : int
        Number of zeros to append.
    axis : int, optional
        Axis along which to pad. Default is `0`.
    where : {'left', 'right', 'both'}, optional
        Where to pad the zeros. Default is `'right'`.

    Returns
    -------
    y : array_like
        Padded array.
    """
    padding = np.zeros((x.ndim, 2), int)
    if where == 'left':
        padding[axis][0] = n
    elif where == 'right':
        padding[axis][1] = n
    elif where == 'both':
        padding[axis][0] = n
        padding[axis][1] = n
    else:
        raise ValueError(f'where must be left, right or both, got {where}')
    return np.pad(x, padding)


def fft_freqs(fs=16e3, n_fft=512, onesided=True):
    """Fast Fourier Transform (FFT) frequency vector.

    Parameters
    ----------
    fs : int or float, optional
        Sampling frequency. Default is `16e3`.
    n_fft : int, optional
        Number of FFT points. Default is `512`.
    onesided : bool, optional
        If `False`, both positive and negative frequencies are returned, which
        corresponds to the output from `np.fft.fft`. Default is `False`, which
        means only positive frequencies are returned; this corresponds to the
        output from `np.fft.rfft`.

    Returns
    -------
    freqs : array_like
        Frequency vector.
    """
    freqs = np.arange(n_fft)*fs/n_fft
    mask = freqs > fs/2
    if onesided:
        freqs = freqs[~mask]
    else:
        freqs[mask] = freqs[mask] - fs
    return freqs


def pretty_table(
    dict_: dict,
    key_header: str = '',
    order_by: str = None,
    reverse: bool = False,
    float_round: int = None,
):
    if not dict_:
        raise ValueError('input is empty')
    # round floats
    if float_round is not None:
        for key, val in dict_.items():
            for subkey, subval in val.items():
                if isinstance(subval, float):
                    dict_[key][subkey] = round(subval, float_round)
    # calculate the first column width
    keys = dict_.keys()
    first_col_width = max(max(len(str(key)) for key in keys), len(key_header))
    # check that all values have the same keys
    values = dict_.values()
    for i, value in enumerate(values):
        if i == 0:
            sub_keys = value.keys()
        elif value.keys() != sub_keys:
            raise ValueError('values in input do not all have same keys')
    # calculate the width of each column
    col_widths = [first_col_width]
    for key in sub_keys:
        col_width = max(max(len(str(v[key])) for v in values), len(key))
        col_widths.append(col_width)
    # define the row formatting
    row_fmt = ' '.join(f'{{:<{width}}} ' for width in col_widths)
    # print the header
    lines_to_print = []
    lines_to_print.append(row_fmt.format(key_header, *sub_keys))
    lines_to_print.append(row_fmt.format(*['-'*w for w in col_widths]))
    # order
    if order_by is None:
        iterator = dict_.items()
    else:
        # type detection
        order_type_cast = float
        for val in dict_.values():
            try:
                float(val[order_by])
            except ValueError:
                order_type_cast = str
                break
        iterator = sorted(
            ((key, val) for key, val in dict_.items()),
            key=lambda x: order_type_cast(x[1][order_by]),
            reverse=reverse
        )
    for key, items in iterator:
        row_fmt = ' '.join(
            f'{{:>{width}}} ' if isinstance(x, (float, int))
            else f'{{:>{width}}} '
            for width, x in zip(col_widths, [key, *items.values()])
        )
        lines_to_print.append(row_fmt.format(key, *items.values()))
    # print lines breaking them into groups if longer than console width
    console_width = shutil.get_terminal_size().columns
    first_col_width += 2
    i_width = 1
    while len(lines_to_print[0]) > first_col_width:
        for i, line in enumerate(lines_to_print):
            end, j_width = first_col_width, i_width
            while j_width < len(col_widths) \
                    and end + col_widths[j_width] + 2 <= console_width:
                end += col_widths[j_width] + 2
                j_width += 1
            print(line[:end])
            lines_to_print[i] = line[:first_col_width] + line[end:]
        i_width = j_width
        print('')
