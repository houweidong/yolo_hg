import os
import sys


# add project root to search path
class context:
    def __enter__(self):
        pro_root = '/home/new/dev/weidong/yolo_hg'
        sys.path.insert(0, pro_root)

    def __exit__(self, *args):
        pass


if __name__ == '__main__':
    print(os.path.realpath('.'))
    print(os.path.dirname(os.path.realpath('.')))
