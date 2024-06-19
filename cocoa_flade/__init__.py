import os
from pathlib import Path
import xml.etree.ElementTree as ET


class FlaDE(object):
    def __init__(self, path) -> None:
        # dataset path
        self.name = 'FlaDE'
        self.path = Path(path)
        if not self.path.exists():
            self.path.mkdir(parents=True)
    
        # download
        if not (self.path / 'assets').exists():
            print("Downloading datasets...")
        #     os.system(f'wget -O {self.name}.zip -P {self.path} '
        #               'TODO: add link')
        #     os.system(f'unzip -q {self.path / self.name}.zip -d {self.path}')
            # os.system(f'rm {self.name}.zip')
        
        # process
        if not (self.path / 'samples').exists():
            print("This is first import, may take some time...")
            os.system(f'python3 {self.path / "scripts" / "build.py"}')

        # parse annotations
        self.root = ET.parse(self.path / 'annotations.xml').getroot()
        self.cats = list()
        self.tags = list()

        # update categories
        self.cats.extend([
            {
                'id': int(label.find('id').text),
                'name': label.find('name').text,
                'color': label.find('color').text
            }
            for label in self.root.findall('meta/labels/label')
        ])
        
        # update aedats
        self.tags.extend([
            {
                # image
                'id': int(image.get('id')),
                'name': image.get('name'),
                'scene': image.get('scene'),
                'frame': image.get('frame'),
                # boxes
                'boxes': [
                    {
                        'label': box.get('label'),
                        'xtl': int(float(box.get('xtl'))),
                        'ytl': int(float(box.get('ytl'))),
                        'xbr': int(float(box.get('xbr'))),
                        'ybr': int(float(box.get('ybr'))),
                    }
                    for box in image.findall('box')
                ],  # 添加逗号
                # group
                'partition': 'train'
            }
            for image in self.root.findall('samples/train/image')
        ])  # part of train
        self.tags.extend([
            {
                # image
                'id': int(image.get('id')),
                'name': image.get('name'),
                'scene': image.get('scene'),
                'frame': image.get('frame'),
                # boxes
                'boxes': [
                    {
                        'label': box.get('label'),
                        'xtl': int(float(box.get('xtl'))),
                        'ytl': int(float(box.get('ytl'))),
                        'xbr': int(float(box.get('xbr'))),
                        'ybr': int(float(box.get('ybr'))),
                    }
                    for box in image.findall('box')
                ],  # 添加逗号
                # group
                'partition': 'test'
            }
            for image in self.root.findall('samples/test/image')
        ])  # part of test

    def _search(self, elements, key='id', query=None):
        # search all
        if query is None:
            return [elem for elem in elements]

        # search by query
        value = []
        for q in query:
            value.extend([
                elem for elem in elements if elem[key] == q
            ])

        return value

    def get_cats(self, key='id', query=None):
        return self._search(self.cats, key=key, query=query)

    def get_tags(self, key='id', query=None):
        return self._search(self.tags, key=key, query=query)


if __name__ == '__main__':
    flade = FlaDE('/home/szd/workspace/tmp-flade/data')
