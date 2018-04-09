right_joints_list = [['RightHand'], 
                     ['RightHand', 'RightForeArm'],
                     ['RightHand', 'RightForeArm', 'RightArm'],
                     ['RightHand', 'RightForeArm', 'RightArm', 'RightShoulder'],
                     ['RightHand', 'RightForeArm', 'RightArm', 'RightShoulder', 'Neck', 'Hips']]

left_joints_list = [['LeftHand'], 
                    ['LeftHand', 'LeftForeArm'],
                    ['LeftHand', 'LeftForeArm', 'LeftArm'],
                    ['LeftHand', 'LeftForeArm', 'LeftArm', 'LeftShoulder'],
                    ['LeftHand', 'LeftForeArm', 'LeftArm', 'LeftShoulder', 'Neck', 'Hips']]

data_types_combination = [['BegMaxEndSpeedNorm'],
                          ['BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz'],
                          ['BegMaxEndSpeedNorm', 'BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz'],
                          ['SpeedNorm'],
                          ['Speedx', 'Speedy', 'Speedz'],
                          ['SpeedNorm', 'Speedx', 'Speedy', 'Speedz'], 
                          ['AccelerationNorm'], 
                          ['Accelerationx', 'Accelerationy', 'Accelerationz'],
                          ['AccelerationNorm', 'Accelerationx', 'Accelerationy', 'Accelerationz'],
                          ['AccelerationNorm', 'SpeedNorm'],
                          ['Accelerationx', 'Accelerationy', 'Accelerationz', 'Speedx', 'Speedy', 'Speedz'],
                          ['AccelerationNorm', 'Accelerationx', 'Accelerationy', 'Accelerationz', 'SpeedNorm', 'Speedx', 'Speedy', 'Speedz']]

people_names = [['Aous', 'left'], ['Damien', 'left'], ['Esteban', 'right'], ['Guillaume', 'right'], 
                ['Ines', 'right'], ['Iza', 'right'], ['Ludovic', 'right'], ['Marc', 'right'],
                ['Oussema', 'right'], ['Pierre', 'right'], ['Sebastien', 'right'], 
                ['Vincent', 'right'], ['Yann', 'right'],
                [['Aous', 'Damien'], 'left'],
                [['Esteban', 'Guillaume', 'Ines', 'Iza', 'Ludovic', 'Marc', 'Oussema', 'Pierre', 'Sebastien', 'Vincent', 'Yann'], 'right']]


joints_name_corres = {'LeftHand': 'LH', 'LeftForeArm': 'LFA', 'LeftArm': 'LA', 'LeftShoulder': 'LS',
                      'RightHand': 'RH', 'RightForeArm': 'RFA', 'RightArm': 'RA', 'RightShoulder': 'RS',
                      'Neck': 'N', 'Hips': 'H'}