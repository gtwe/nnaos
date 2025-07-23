import os
import sys

import numpy as np
import pandas as pd

sys.path.append("../../")


def rates(dofs: np.array, errors: np.array) -> np.array:
    """
    Numerical estimate of convergence rates
    """
    assert len(dofs) == len(errors)

    rates = np.empty(len(dofs))
    rates[0] = np.nan
    rates[1:] = -np.log(errors[1:] / errors[:-1]) / np.log(dofs[1:] / dofs[:-1])

    return rates


def df_to_tex(df, name, expname):

    with open(f'../../reports/{expname}/tex/{name}.tex', 'w') as f:

        df.index.name = None
        table_str = df.style.format(na_rep='').to_string(delimiter=' & ')
        table_str = table_str.replace('\n', '\\\\\n')

        # Remove last 'newline'
        table_str = table_str[:-3]
        table_str = table_str[len("samples") :]

        # Add newline '\\' for latex
        table_str = table_str.replace('\n', ' \\hline\n', 1)
        f.write(table_str)


def csv_maker(p1, p2, expname):
    print('Making csv files')
    """ Generates csv files for the tables in the paper

    Args:
        p1: first parameter, calculate rate of test loss vs. this param
        p2: second parameter, calculate rate of test loss vs. this param
    """
    df = pd.read_pickle("./results/df.pkl")

    df['width'] = df.apply(lambda row: row.params.module_args.width, axis=1)

    savedir = f"../../reports/{expname}/csv/"

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    groupby = ["id", p1, p2, "dim"]
    data = df.groupby(groupby, as_index=False).aggregate(
        {
            'id': 'first',
            p1: 'first',
            p2: 'first',
            'dof': 'first',
            'width': 'first',
            'dim': 'first',
            'run': 'first',
            'test_loss': 'mean',
        }
    )

    dfs = []
    for samples, group in data.groupby(p2):
        group = group.sort_values(p1)
        group[f'{p1}_rate'] = rates(group[p1].to_numpy(), group['test_loss'].to_numpy())
        dfs.append(group)
    p1_data = pd.concat(dfs)

    p1_data = p1_data.sort_index()
    data[f'{p1}_rate'] = p1_data[f'{p1}_rate']

    dfs = []
    for sample, group in data.groupby(p1):
        group = group.sort_values(p2)
        group[f'{p2}_rate'] = rates(group[p2].to_numpy(), group['test_loss'].to_numpy())
        dfs.append(group)
    p2_data = pd.concat(dfs)

    p2_data = p2_data.sort_index()
    data[f'{p2}_rate'] = p2_data[f'{p2}_rate']

    p1_rates = p1_data.pivot(index=p1, columns=p2, values=f'{p1}_rate')
    p2_rates = p2_data.pivot(index=p1, columns=p2, values=f'{p2}_rate')
    test_loss = p2_data.pivot(index=p1, columns=p2, values='test_loss')

    p1_rates.index.name = f'{p1}\\textbackslash {p2}'
    p2_rates.index.name = f'{p1}\\textbackslash {p2}'
    test_loss.index.name = f'{p1}\\textbackslash {p2}'

    p1_rates = p1_rates.tail(4).iloc[:, -4:]
    p2_rates = p2_rates.tail(4).iloc[:, -4:]
    test_loss = test_loss.tail(4).iloc[:, -4:]

    p1_rates.to_csv(f"{savedir}{p1}_rates_by_{p1}.csv")
    p2_rates.to_csv(f"{savedir}{p2}_rates_by_{p1}.csv")
    test_loss.to_csv(f"{savedir}{p1}_test_loss_by_{p1}.csv")

    p1_rates = p1_rates.map(lambda x: f'{x: .3}').astype(str)
    p2_rates = p2_rates.map(lambda x: f'{x: .3}').astype(str)
    test_loss = test_loss.map(lambda x: f'{x: .3}').astype(str)

    p1_rates.to_csv(f"{savedir}{p1}_rates_rounded_by_{p1}.csv")
    p2_rates.to_csv(f"{savedir}{p2}_rates_rounded_by_{p1}.csv")
    test_loss.to_csv(f"{savedir}{p1}_test_loss_rounded_by_{p1}.csv")

    df_to_tex(p1_rates, f'{p1} rates by {p1}', expname)
    df_to_tex(p2_rates, f'{p2} rates by {p1}', expname)
    df_to_tex(test_loss, f'{p1} test loss by {p1}', expname)


def read_tex_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    return lines


def write_tex_file(filename, lines):
    with open(filename, 'w') as file:
        file.writelines(lines)


def make_quad(p1, p2, expname):
    input_file1 = f'../../reports/{expname}/tex/{p1} rates by {p1}.tex'
    input_file2 = f'../../reports/{expname}/tex/{p2} rates by {p1}.tex'
    output_file = f'../../reports/{expname}/tex/{p1} {p2} quad.tex'

    lines1 = read_tex_file(input_file1)
    lines2 = read_tex_file(input_file2)

    lines1 = [line.split('\\', 1)[0] + '' for line in lines1]

    lines2 = ['&' + line.split('&', 1)[1] for line in lines2]

    output_lines = []

    for i in range(len(lines1)):
        output_lines.append(lines1[i] + lines2[i])

    with open(output_file, 'w') as f:
        f.writelines(output_lines)


def make_row(p1, p2, expname):
    print('make row')
    input_file1 = f'../../reports/{expname}/tex/{p1} test loss by {p1}.tex'
    input_file2 = f'../../reports/{expname}/tex/{p2} rates by {p1}.tex'
    input_file3 = f'../../reports/{expname}/tex/{p2} rates by {p1}.tex'
    output_file = f'../../reports/{expname}/tex/{p1} {p2} row.tex'

    lines1 = read_tex_file(input_file1)
    lines2 = read_tex_file(input_file2)
    lines3 = read_tex_file(input_file3)

    lines1 = [line.split('\\', 1)[0] + '' for line in lines1]

    lines2 = ['&' + line.split('&', 1)[1] for line in lines2]

    lines2 = [line.split('\\', 1)[0] + '' for line in lines2]

    lines3 = ['&' + line.split('&', 1)[1] for line in lines3]

    output_lines = []

    for i in range(len(lines1)):
        output_lines.append(lines1[i] + lines2[i] + lines3[i])

    with open(output_file, 'w') as f:
        f.writelines(output_lines)


def make_tex_files(expname):
    savedir = f'../../reports/{expname}/tex/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    p1, p2 = 'width', 'samples'

    print(f'Generating {expname} tables {p1} and {p2}')
    csv_maker(p1, p2, expname)
    make_quad(p1, p2, expname)
    make_row(p1, p2, expname)

    p1, p2 = 'dof', 'samples'

    print(f'Generating {expname} tables {p1} and {p2}')
    csv_maker(p1, p2, expname)
    make_quad(p1, p2, expname)
    make_row(p1, p2, expname)
