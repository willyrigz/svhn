
# here I will import the main module from your code - you need to make sure your code imports without a problem
# As per the assignment specification, your main module must be called svhn.py
import svhn

def main():

    # I might start by calling on your code to do some processing based on the model that you already trained
    result1 = svhn.test("test_images/myhouse.jpeg")
    print(result1)

    result2 = svhn.test("test_images/9852.jpg")
    print(result2)

    result3 = svhn.test("test_images/r1c.jpg")
    print(result3)

    # i might also test with a PNG
    result4 = svhn.test("test_images/1.png")
    print(result4)

    result5 = svhn.test("test_images/5004.png")
    print(result5)

    result6 = svhn.test("test_images/Capture2.png")
    print(result6)

    # I will also call to start training on your code from scratch. I might not always wait for training to complete
    # but I will start the training and make sure it is progressing.
    average_f1_scores = svhn.traintest()
    print(average_f1_scores)


if __name__ == '__main__':
    main()