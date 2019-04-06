import os

arrAlphas = ["0.1", "0.05", "0.01"]
arrExplorationTypes = ["e-greedy", "softmax"]
arrMaps = ["MiniGrid-Empty-8x8-v0"]
arrC = [0.9, 0.5, 0.1]



for map in arrMaps:
    for explorationType in arrExplorationTypes:
        for alpha in arrAlphas:
            for c in arrC:
                os.system("time python3 sarsa_skel.py --environment " + str(map) + " --strategy "+ str(explorationType) +" --epochs 50000  --c " + str(c) +" --learning_rate " + str(alpha))
