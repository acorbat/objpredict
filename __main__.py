import argparse

import process_experiment as exp

def list_installed_pipelines(args):
    for p in exp.get_installed_pipelines().keys():
        if p.startswith('./'):
            print(p + ' (local)')
        else:
            print(p + ' (global)')


def _install(namespace):
    return exp.install(namespace.pipeline, namespace.name)


def _process_experiment(namespace):
    return exp.process_experiment(namespace.filename, namespace.append, namespace.pipeline)


def _train(namespace):
    return exp.train(namespace.folder, namespace.output)


parser = argparse.ArgumentParser(prog='objpredict', description='Classify objects')

subparsers = parser.add_subparsers()

p_classif = subparsers.add_parser('predict', help='Classify objects from an objects CellProfiler *_Image.csv file.')
p_classif.add_argument('filename', nargs='+')
p_classif.add_argument('-p', '--pipeline')
p_classif.add_argument('-a', '--append', default=False)
p_classif.set_defaults(func=_process_experiment)

l_classif = subparsers.add_parser('list', help='List installed pipelines.')
l_classif.set_defaults(func=list_installed_pipelines)

u_classif = subparsers.add_parser('update', help='Update pipelines.')
u_classif.set_defaults(func=list_installed_pipelines)

t_classif = subparsers.add_parser('train', help='Train classifier.')
t_classif.add_argument('folder', nargs='+')
t_classif.add_argument('-o', '--output', default='MyPipeline')
t_classif.set_defaults(func=_train)

s_classif = subparsers.add_parser('install', help='Install pipelines.')
s_classif.add_argument('pipeline')
s_classif.add_argument('name', default=None)
s_classif.set_defaults(func=_install)


args = parser.parse_args()

if not hasattr(args, 'func'):
    parser.print_help()
else:
    args.func(args)
