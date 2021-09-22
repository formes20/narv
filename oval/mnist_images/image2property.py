##   7,2,1,0,4,1,4,9,5,9,0,6,9,0,1,5,9,7,3,4,9,6,6,5,4,   17,18,19,20,21,22,23,24,25
def property2line(label_list, length):
    
    for i in range(length):
        with open('image'+str(i),'r') as f:
            pixels = f.readline()
        pixels = list(pixels.split(','))
        #print(pixels)
        delta = 0.01
        index = 0
        print(f'"adversarial_{str(i)}":', end=" ")
        print(r"{")
        # input
        print(f'\t"type": "adversarial",')
        print(f'\t"input":')
        print('\t\t[')
        for pixel in pixels:
            if pixel != "":
                pixel_val = float(pixel)/255
                if pixel_val <= delta:
                    pixel_lower = 0
                else:
                    pixel_lower = pixel_val - delta
                pixel_upper = pixel_val + delta
                if pixel_upper > 1:
                    pixel_upper = 1
                print(f'\t\t\t({index},', end=" ")
                print(r"{", end="")
                print(f'"Lower": {pixel_lower}, "Upper": {pixel_upper}', end="")
                print(r"}),")
                index += 1
        print(f'\t\t],')
        print(f'\t"output":')
        print('\t\t[')
        for j in range(10):
            print(f'\t\t\t({j},', end=" ")
            print(r"{", end="")
            if j == label_list[i-1]:
                print(f'"Lower": {-1}, "Upper": {0}', end="")
            else:    
                print(f'"Lower": {0}, "Upper": {0}', end="")
            print(r"}),")
        print(f'\t\t]')
        print(r"},")