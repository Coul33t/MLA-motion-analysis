import frame

class Motion():
    def __init__(self, name='NONE', frame_time=0):
        self.name = name
        self.frame_time = frame_time
        self.frames = []

    def interpolate_joint(self, j1, j2, factor):
        pass

    def interpolate_frames(self, f1, f2, factor):
        pass

    def parse_bvh(path_to_file):
        is_data = False

        with open(path_to_file, 'r') as data:
            text = data.read()

            for line in enumerate(text.split('\n')):
                splitted_line = line[1].split()

                if(splitted_line):
                    if(splitted_line[0] == 'Frame'):
                        is_data = True

                    if not is_data:
                        pass
