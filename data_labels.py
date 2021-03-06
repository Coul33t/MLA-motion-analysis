# Natural index (1 -> first element, opposed to array idx where 0 -> first element)
GLOUP_SUCCESS = [1, 8, 19, 20, 22, 24, 25, 28, 32, 33, 39, 40, 47, 56, 57, 60, 73, 74, 77, 79, 83, 84, 95, 99, 100]
DBRUN_SUCCESS = [2, 6, 10, 14, 15, 18, 21, 27, 29, 30, 44, 46, 48, 50, 59, 63, 64, 65, 69, 70, 71, 73, 74, 76, 77, 83, 86, 87, 88, 89, 90, 93, 97, 100]

                  ##############
                  # BFC LABELS #
                  ##############
GLOUP_LABELS_2 = [1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                  0, 1, 0, 1, 1, 0, 0, 1, 0, 0,
                  0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
                  0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                  0, 0, 0, 0, 0, 1, 1, 0, 0, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 1, 1, 0, 0, 1, 0, 1, 0,
                  0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 1, 0, 0, 0, 1, 1]

DBRUN_LABELS_2 = [0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
                  0, 0, 0, 1, 1, 0, 0, 1, 0, 0,
                  1, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                  0, 0, 1, 1, 1, 0, 0, 0, 1, 1,
                  1, 0, 1, 1, 0, 1, 1, 0, 0, 0,
                  0, 0, 1, 0, 0, 1, 1, 1, 1, 1,
                  0, 0, 1, 0, 0, 0, 1, 0, 0, 1]

ELOISEAU_LABELS_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
                     0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
                     1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 0, 0, 0]


VBETTENFELD_LABELS_2 = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
                        0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
                        0, 0, 0, 0, 1, 0, 1, 1, 0, 0,
                        0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                        0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                        1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

IMARFISI_LABELS_2 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
                     0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

MLECONTE_LABELS_2 = [0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
                     1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                     1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
                     1, 0, 0, 1, 1, 1, 0, 1, 0, 0]

PLAFORCADE_LABELS_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                       1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                       1, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                       0, 1, 1, 0, 1, 1, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                       0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                       0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
                       0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                       0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0,]

OMAHDI_LABELS_2 = [0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 1, 0, 0, 1, 1, 0,
                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                   0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
                   0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

AKARAOUI_LABELS_2 = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                     0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                     1, 0, 0, 0, 0, 1, 0, 0, 0, 1,
                     0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                     1, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                     0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
                     0, 0, 1, 1, 1, 0, 1, 1, 0, 1,
                     0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                     1, 1, 0, 0, 1, 0, 0, 0, 0, 1,
                     0, 1, 1, 0, 1, 1, 0, 0, 1, 0]

YWALKOWIAK_LABELS_2 = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
                       0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                       1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                       0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
                       0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 1, 0, 0, 1, 1,
                       0, 1, 1, 0, 0, 0, 0, 1, 1, 0,
                       1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                       0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

LHAMON_LABELS_2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

SGEORGE_LABELS_2 = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                    0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                    0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
                    0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                    0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

IDABBEBI_LABELS_2 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                     0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                     0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
                     0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                     0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
                     0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

            ########################
            # ALTERNATIVE LABELING #
            ########################

# Same as DRUN_LABELS_4, except the 3 have been put to 2
DBRUN_LABELS_3_1 = [0, 1, 0, 0, 2, 1, 0, 2, 2, 1,
                    0, 0, 0, 1, 1, 2, 2, 1, 0, 2,
                    1, 0, 0, 0, 2, 2, 1, 0, 1, 1,
                    0, 2, 2, 0, 0, 0, 0, 2, 0, 0,
                    0, 0, 0, 1, 0, 1, 0, 1, 2, 1,
                    0, 2, 0, 0, 2, 0, 2, 0, 1, 0,
                    2, 2, 1, 1, 1, 2, 2, 0, 1, 1,
                    1, 2, 1, 1, 0, 1, 1, 0, 0, 0,
                    0, 0, 1, 0, 2, 1, 1, 1, 1, 1,
                    2, 2, 1, 2, 0, 0, 1, 0, 0, 1]

# Same as DRUN_LABELS_4, except the 3 have been put to 0
DBRUN_LABELS_3_2 = [0, 1, 0, 0, 0, 1, 0, 2, 2, 1,
                    0, 0, 0, 1, 1, 2, 0, 1, 0, 0,
                    1, 0, 0, 0, 2, 0, 1, 0, 1, 1,
                    0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 2, 1, 1, 1, 0, 0, 0, 1, 1,
                    1, 2, 1, 1, 0, 1, 1, 0, 0, 0,
                    0, 0, 1, 0, 2, 1, 1, 1, 1, 1,
                    2, 0, 1, 0, 0, 0, 1, 0, 0, 1]

# 0 = fail, 1 = success, 2 = almost, 3 = not sure (can merge 2 and 3 OR exclude 3)
DBRUN_LABELS_4 = [0, 1, 0, 0, 3, 1, 0, 2, 2, 1,
                  0, 0, 0, 1, 1, 2, 3, 1, 0, 3,
                  1, 0, 0, 0, 2, 3, 1, 0, 1, 1,
                  0, 2, 3, 0, 0, 0, 0, 3, 0, 0,
                  0, 0, 0, 1, 0, 1, 0, 1, 3, 1,
                  0, 3, 0, 0, 3, 0, 3, 0, 1, 0,
                  3, 2, 1, 1, 1, 3, 3, 0, 1, 1,
                  1, 2, 1, 1, 0, 1, 1, 0, 0, 0,
                  0, 0, 1, 0, 2, 1, 1, 1, 1, 1,
                  2, 3, 1, 3, 0, 0, 1, 0, 0, 1]

# SMALL TEST LABELS

MOVE_SUCCESS = [x for x in range(1, 11)]
THROW_SUCCESS = [x for x in range(1, 11)]
SQUARE_SUCCESS = [x for x in range(1, 11)]


# THROW BALL LABELS

# Success or fail
LEO_LABELS_2 = [0, 1, 0, 1, 1, 1, 1, 0, 1, 0,
                1, 1, 0, 1, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
                1, 0, 0, 1, 1, 1, 0, 1, 0, 1,
                1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                1, 0, 0, 1, 1, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 1, 1, 1, 0, 0, 1]

# 0 = Close bin
# 1 = Far bin
LEO_THROW_LABELS = [0 for x in range(100)]
LEO_THROW_LABELS = [x if i > 49 else 1 for i,x in enumerate(LEO_THROW_LABELS)]

# 0 = Up
# 1 = Down
LEO_THROW_TYPES = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
                   1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                   0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                   1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


# DARTS
FAKE_DARTS_LABELS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                     3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

AURELIEN_LABELS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                   4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

MIXED_LABELS =    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   2]