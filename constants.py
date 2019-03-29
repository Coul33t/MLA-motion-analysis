implemented_algo = ('k-means', 'dbscan', 'agglomerative', 'mean-shift', 'gmm')

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

neutral_joints_list = [['Hand'],
                      ['Hand', 'ForeArm'],
                      ['Hand', 'ForeArm', 'Arm'],
                      ['Hand', 'ForeArm', 'Arm', 'Shoulder'],
                      ['Hand', 'ForeArm', 'Arm', 'Shoulder', 'Neck', 'Hips']]

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

people_names_O = [['Aous', 'left'], ['Damien', 'left'], ['Esteban', 'right'], ['Guillaume', 'right'],
                ['Ines', 'right'], ['Iza', 'right'], ['Ludovic', 'right'], ['Marc', 'right'],
                ['Oussema', 'right'], ['Pierre', 'right'], ['Sebastien', 'right'],
                ['Vincent', 'right'], ['Yann', 'right'],
                [['Aous', 'Damien'], 'left'],
                [['Esteban', 'Guillaume', 'Ines', 'Iza', 'Ludovic', 'Marc', 'Oussema', 'Pierre', 'Sebastien', 'Vincent', 'Yann'], 'right']]


people_names = [['Aous', 'left'], ['Damien', 'left'], ['Esteban', 'right'], ['Guillaume', 'right'],
                ['Ines', 'right'], ['Iza', 'right'], ['Ludovic', 'right'], ['Marc', 'right'],
                ['Oussema', 'right'], ['Pierre', 'right'], ['Sebastien', 'right'],
                ['Vincent', 'right'], ['Yann', 'right']]

names_labels = {'Aous': 'AKARAOUI_LABELS_2',
                'Damien': 'DBRUN_LABELS_2',
                'Esteban': 'ELOISEAU_LABELS_2',
                'Guillaume': 'GLOUP_LABELS_2',
                'Ines': 'IDABBEBI_LABELS_2',
                'Iza': 'IMARFISI_LABELS_2',
                'Ludovic': 'LHAMON_LABELS_2',
                'Marc': 'MLECONTE_LABELS_2',
                'Oussema': 'OMAHDI_LABELS_2',
                'Pierre': 'PLAFORCADE_LABELS_2',
                'Sebastien': 'SGEORGE_LABELS_2',
                'Vincent': 'VBETTENFELD_LABELS_2',
                'Yann': 'YWALKOWIAK_LABELS_2'}

joints_name_corres = {'LeftHand': 'LH', 'LeftForeArm': 'LFA', 'LeftArm': 'LA', 'LeftShoulder': 'LS',
                      'RightHand': 'RH', 'RightForeArm': 'RFA', 'RightArm': 'RA', 'RightShoulder': 'RS',
                      'Hand': 'H', 'ForeArm': 'FA', 'Arm': 'A', 'Shoulder': 'S',
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


problems_and_advices = {'leaning': "Your shoulders shouldn't move when you're throwing.",
                        'javelin': "Your throwing hand should always be in front of you when you're throwing.",
                        'align_arm': "Your arm should stay aligned regarding to your body.",
                        'elbow_move': "Your elbow shouldn't move while you're throwing."}

problemes_et_solutions = {'leaning': "Votre corps et vos épaules ne doivent pas bouger pendant votre lancer.",
                          'javelin': "Votre main doit toujours rester devant votre corps lorsque vous lancez.",
                          'align_arm': "Votre bras doit rester aligné (de la main à l'épaule) lorsque vous lancez.",
                          'elbow_move': "Votre coude ne doit pas bouger lors du lancer."}