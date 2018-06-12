from prototype.Model import *
import cv2

model = Model("tiles/paths", (32, 32), 3)
model.generate_image()

if False:
    break_all = False
    print(model.patterns[6][0,0])
    print()

    for p_i_1 in range(model.num_patterns):
        scaled1 = cv2.resize(model.patterns[p_i_1], (128, 128), interpolation=cv2.INTER_AREA)
        for p_i_2 in range(model.num_patterns):
            scaled2 = cv2.resize(model.patterns[p_i_2], (128, 128), interpolation=cv2.INTER_AREA)
            comb = np.hstack((scaled1, np.full((128, 10, 3), 128), scaled2))
            print("template: {}, conv: {}, result:\n{}\n".format(p_i_1, p_i_2, model.fit_table[p_i_1,p_i_2]))
            cv2.imshow("comparison", comb/255.0)
            k = cv2.waitKey(0)
            if k == 27:
                break_all = True
                break
            elif k == ord('n'):
                break
        if break_all:
            break

result = cv2.resize(model.out_img, (800, 800), interpolation=cv2.INTER_AREA)
# result = model.out_img
cv2.imshow("result", result/255.0)
cv2.waitKey(0)
cv2.imwrite("result.png", result)

# sub = np.array([[-1, -1, -1],
#                 [-1, -1,  1],
#                 [-1,  1,  1]])
#
# obsv = np.array([[False, False, False],
#                  [False, False,  True],
#                  [False,  True,  True]])
#
# pattern = np.array([[ 0,  0,  1],
#                     [ 0,  1,  1],
#                     [ 1,  1,  1]])
#
# matches = np.equal(sub, pattern)
# obsv_m = np.array_equal(matches, obsv)
#
# print(matches)
# print(obsv_m)
