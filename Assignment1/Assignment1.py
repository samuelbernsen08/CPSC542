import program

repeat = True
while(repeat):
    train = input("Would you like to train the network? Enter Y for yes and N for no: ")
    if train == "Y" or train == "y" or train == "Yes" or train == "yes":
        condition = True
        break
    elif train == "N" or train == "n" or train == "No" or train == "no":
        condition = False
        break
    else:
        repeat = True
        print("Invalid input")
        print()

program.run(condition)
