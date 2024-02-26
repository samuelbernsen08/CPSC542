import program

train = input("Would you like to train the network? Enter Y for yes and N for no: ")
if train == "Y" or train == "y" or train == "Yes" or train == "yes":
    condition = True
else:
    condition = False
    

program.run(condition)
