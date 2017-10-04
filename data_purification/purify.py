import pdb

def test(folder_path, filename):
    with open(folder_path+'\\'+filename+'.bvh', 'r') as file_data, open(folder_path+'\\'+filename+'_pur.bvh', 'w') as output_file:
        text = file_data.read()

        for line in enumerate(text.split('\n')):
            output_file.write(line[1] + '\n')


def purify(folder_path, filename):

    number_of_channels_for_ref = 0
    wait_for_joint = False
    pos_and_ori = False
    DEBUG = False

    with open(folder_path+'\\'+filename+'.bvh', 'r') as file_data, open(folder_path+'\\'+filename+'_pur.bvh', 'w') as output_file:
        text = file_data.read()

        for line in enumerate(text.split('\n')):
            splitted_line = line[1].split(' ')

            if splitted_line:
                # Still at the offset part
                if not pos_and_ori:

                    # no reference joint yet
                    if not wait_for_joint:


                        # if it's the reference node, stop writing
                        if splitted_line[0] == 'ROOT' and splitted_line[1] == 'reference':
                            output_file.write(splitted_line[0])
                            wait_for_joint = True
                        
                        # if it's the position and orientation part, stop going into this part of the code
                        elif splitted_line[0] =='Frame':
                            pos_and_ori = True
                            output_file.write(line[1] + '\n')
                        
                        # if it's the additionnal '}'
                        elif splitted_line[0] == '}' and len(splitted_line) == 1:
                            pass
                        
                        # else, write the line
                        else:
                            output_file.write(line[1] + '\n')

                    # else, wait for the next joint (true root)
                    else:
                        if 'CHANNELS' in splitted_line:
                            number_of_channels_for_ref = int(splitted_line[splitted_line.index('CHANNELS') + 1])

                        if 'JOINT' in splitted_line:
                            wait_for_joint = False
                            output_file.write(' ' + splitted_line[splitted_line.index('JOINT') + 1] + '\n')

                else:
                    for i in range(number_of_channels_for_ref, len(splitted_line)):
                        output_file.write(splitted_line[i] + ' ')

                    output_file.write('\n');
                

def main():
    folder_path = 'C:\\Users\\quentin\\Documents\\Programmation\\Python\\ml_mla'
    filename = 'throw_1_cut'
    purify(folder_path, filename)
    #test(folder_path, filename)

if __name__ == '__main__':
    main()