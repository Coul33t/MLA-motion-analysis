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
                          ['BegMaxEndSpeedx',     'BegMaxEndSpeedy',    'BegMaxEndSpeedz'],
                          ['BegMaxEndSpeedDirx',  'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz'],
                          ['BegMaxEndSpeedNorm',  'BegMaxEndSpeedx',    'BegMaxEndSpeedy',    'BegMaxEndSpeedz'],
                          ['BegMaxEndSpeedNorm',  'BegMaxEndSpeedDirx', 'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz'],
                          ['BegMaxEndSpeedDirx',  'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz', 'BegMaxEndSpeedx',    'BegMaxEndSpeedy', 'BegMaxEndSpeedz'],
                          ['BegMaxEndSpeedNorm',  'BegMaxEndSpeedDirx', 'BegMaxEndSpeedDiry', 'BegMaxEndSpeedDirz', 'BegMaxEndSpeedx', 'BegMaxEndSpeedy', 'BegMaxEndSpeedz'],

                          # ['SpeedNorm'],
                          # ['Speedx',    'Speedy',     'Speedz'],
                          # ['SpeedDirx', 'SpeedDiry',  'SpeedDirz'],
                          # ['SpeedNorm', 'Speedx',     'Speedy',     'Speedz'],
                          # ['SpeedNorm', 'SpeedDirx',  'SpeedDiry',  'SpeedDirz'],
                          # ['Speedx',    'Speedy', 'Speedz', 'SpeedDirx',  'SpeedDiry', 'SpeedDirz'],
                          # ['SpeedNorm', 'Speedx', 'Speedy', 'Speedz',     'SpeedDirx', 'SpeedDiry', 'SpeedDirz'],

                          # ['AccelerationNorm'],
                          # ['Accelerationx',     'Accelerationy',    'Accelerationz'],
                          # ['AccelerationDirx',  'AccelerationDiry', 'AccelerationDirz'],
                          # ['AccelerationNorm', 'Accelerationx',     'Accelerationy',    'Accelerationz'],
                          # ['AccelerationNorm', 'AccelerationDirx',  'AccelerationDiry', 'AccelerationDirz'],
                          # ['Accelerationx',     'Accelerationy', 'Accelerationz', 'AccelerationDirx',   'AccelerationDiry',   'AccelerationDirz'],
                          # ['AccelerationNorm',  'Accelerationx', 'Accelerationy', 'Accelerationz',      'AccelerationDirx',   'AccelerationDiry', 'AccelerationDirz'],

                          # ['AccelerationNorm', 'SpeedNorm'],
                          # ['AccelerationNorm', 'BegMaxEndSpeedNorm'],
                          # ['SpeedNorm', 'BegMaxEndSpeedNorm'],
                          # ['AccelerationNorm', 'SpeedNorm', 'BegMaxEndSpeedNorm']
                          ]

people_names = [['Aous', 'left'], ['Damien', 'left'], ['Esteban', 'right'], ['Guillaume', 'right'],
                ['Ines', 'right'], ['Iza', 'right'], ['Ludovic', 'right'], ['Marc', 'right'],
                ['Oussema', 'right'], ['Pierre', 'right'], ['Sebastien', 'right'],
                ['Vincent', 'right'], ['Yann', 'right'],
                [['Aous', 'Damien'], 'left'],
                [['Esteban', 'Guillaume', 'Ines', 'Iza', 'Ludovic', 'Marc', 'Oussema', 'Pierre', 'Sebastien', 'Vincent', 'Yann'], 'right']]

joints_name_corres = {'LeftHand': 'LH', 'LeftForeArm': 'LFA', 'LeftArm': 'LA', 'LeftShoulder': 'LS',
                      'RightHand': 'RH', 'RightForeArm': 'RFA', 'RightArm': 'RA', 'RightShoulder': 'RS',
                      'Neck': 'N', 'Hips': 'H'}

data_types_corres = {'BegMaxEndSpeedNorm_BegMaxEndSpeedx_BegMaxEndSpeedy_BegMaxEndSpeedz': 'BegMaxEndSpeedNormxyz',
                     'BegMaxEndSpeedNormBegMaxEndSpeedxBegMaxEndSpeedyBegMaxEndSpeedz': 'BegMaxEndSpeedNormxyz',

                     'BegMaxEndSpeedx_BegMaxEndSpeedy_BegMaxEndSpeedz': 'BegMaxEndSpeedxyz',
                     'BegMaxEndSpeedxBegMaxEndSpeedyBegMaxEndSpeedz': 'BegMaxEndSpeedxyz',

                     'BegMaxEndSpeedDirx_BegMaxEndSpeedDiry_BegMaxEndSpeedDirz': 'BegMaxEndSpeedDirxyz',
                     'BegMaxEndSpeedDirxBegMaxEndSpeedDiryBegMaxEndSpeedDirz': 'BegMaxEndSpeedDirxyz',

                     'AccelerationNorm_Accelerationx_Accelerationy_Accelerationz': 'AccelerationNormxyz',
                     'AccelerationNormAccelerationxAccelerationyAccelerationz': 'AccelerationNormxyz',

                     'Accelerationx_Accelerationy_Accelerationz': 'Accelerationxyz',
                     'AccelerationxAccelerationyAccelerationz': 'Accelerationxyz',

                     'SpeedDirx_SpeedDiry_SpeedDirz': 'SpeedDirxyz',
                     'SpeedDirxSpeedDirySpeedDirz': 'SpeedDirxyz',

                     'AccelerationDirx_AccelerationDiry_AccelerationDirz': 'AccDirxyz',
                     'AccelerationDirxAccelerationDiryAccelerationDirz': 'AccDirxyz',

                     'Dirx_Diry_Dirz': 'Dirxyz',
                     'DirxDiryDirz': 'Dirxyz',

                     'Speedx_Speedy_Speedz': 'Speedxyz',
                     'SpeedxSpeedySpeedz': 'Speedxyz',

                     'BegMaxEndSpeed': 'BMES',
                     'Acceleration': 'Acc'}

data_types_base_name = ['BegMaxEndSpeed', 'Speed', 'Acceleration']