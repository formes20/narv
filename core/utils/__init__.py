

# import pandas as pd
# import matplotlib.pyplot as plt
#
# # extract marabou_before_pp results
# # df_results_marabou_before_pp = pd.read_json("/home/yizhak/Desktop/compare_results_by_property/medium/marabou_before_pp_2019-10-23/outfile_df2json/df_all")
# # df_results_marabou_before_pp_l500 = df_results_marabou_before_pp[df_results_marabou_before_pp.filenames.str.contains("L_500")]
# # df_results_marabou_before_pp_l100 = df_results_marabou_before_pp[df_results_marabou_before_pp.filenames.str.contains("L_100")]
# # for row in df_results_marabou_before_pp_l500.iterrows():
# #     print(row[1]["net name"], float(row[1]["orig_query_time"]))
# # for row in df_results_marabou_before_pp_l100.iterrows():
# #     print(row[1]["net name"], float(row[1]["orig_query_time"]))
# #
# # d_marabou_before_pp = {
# #     # 500
# #     "2_7 500": 1.0159204006,
# #     "4_7 500": 14.5495357513,
# #     "2_2 500": 326.8431725502,
# #     "1_8 500": 0.1554965973,
# #     "3_1 500": 1.9661562443,
# #     "1_2 500": 7.1116273403,
# #     "5_9 500": 18.0527923107,
# #     "3_8 500": 2.6356737614,
# #     "1_6 500": 0.134771347,
# #     "5_4 500": 1.9356229305000001,
# #     "3_3 500": 1.8027741909000001,
# #     "1_9 500": 0.1717176437,
# #     "2_1 500": 1.379750967,
# #     "5_1 500": 28.1365590096,
# #     "5_7 500": 64.0679690838,
# #     "3_2 500": 14.6771950722,
# #     "2_9 500": 1.0880796909,
# #     "2_3 500": 1.0949096680000001,
# #     "5_6 500": 2.8383886814,
# #     "1_3 500": 21.4204211235,
# #     "5_5 500": 1.2511730194,
# #     "4_2 500": 28.9906408787,
# #     "1_4 500": 1.9712622166,
# #     "1_7 500": 0.1462953091,
# #     "4_4 500": 80.005995512,
# #     "5_3 500": 13.6753280163,
# #     "4_8 500": 173.5794382095,
# #     "3_5 500": 3.0112342834,
# #     "2_4 500": 1.3021833897000001,
# #     "5_8 500": 112.3851261139,
# #     "1_1 500": 6.4849009514,
# #     "2_5 500": 17.7461600304,
# #     "2_6 500": 12.0707089901,
# #     "4_9 500": 1.1836915016,
# #     "2_8 500": 1.305015564,
# #     "3_9 500": 28.3047792912,
# #     "5_2 500": 1.2671105862,
# #     "3_4 500": 7.2207305431,
# #     "4_1 500": 4.8010663986,
# #     "1_5 500": 1.1100697517,
# #     "4_3 500": 14.9511222839,
# #     "3_7 500": 81.7271232605,
# #     "4_5 500": 1.0740561485,
# #     "4_6 500": 75.6279530525,
# #     "3_6 500": 128.5801711082,
# #     # 100
# #     "3_8 100": 17.0393908024,
# #     "1_5 100": 1.936435461,
# #     "4_9 100": 13.4525825977,
# #     "4_6 100": 40.7014899254,
# #     "3_5 100": 3.4315354824,
# #     "1_1 100": 5.6763412952,
# #     "5_7 100": 1.8136997223,
# #     "4_2 100": 7.875546217,
# #     "4_3 100": 49.5829443932,
# #     "2_7 100": 21.7395262718,
# #     "4_4 100": 20.872661829,
# #     "2_9 100": 14.8920886517,
# #     "4_8 100": 64.5803861618,
# #     "3_9 100": 1.1238877773,
# #     "4_5 100": 6.3976006508,
# #     "3_3 100": 32.2100167274,
# #     "1_4 100": 4.7982573509,
# #     "2_1 100": 64.500597477,
# #     "5_4 100": 5.0359609127,
# #     "3_6 100": 2.2476360798,
# #     "2_4 100": 4.6240718365,
# #     "1_2 100": 101.5314514637,
# #     "4_7 100": 11.5435135365,
# #     "3_7 100": 25.2513995171,
# #     "3_4 100": 349.4756522179,
# #     "2_8 100": 17.2554121017,
# #     "5_2 100": 140.3676254749,
# #     "4_1 100": 145.7993659973,
# #     "2_6 100": 5.5763614178,
# #     "5_8 100": 104.1567203999,
# #     "2_5 100": 1.1838755608,
# #     "5_3 100": 5.6992759705000005,
# #     "1_9 100": 0.0997107029,
# #     "1_8 100": 17.1193385124,
# #     "5_5 100": 146.434458971,
# #     "2_2 100": 1405.9914908409,
# #     "2_3 100": 5.6657431126,
# #     "5_9 100": 134.6164839268,
# #     "1_7 100": 86.7406995296,
# #     "5_1 100": 59.4701883793,
# #     "5_6 100": 1.3925311565,
# #     "1_6 100": 14.834127903,
# #     "1_3 100": 1.8508007526,
# #     "3_2 100": 20.7987277508,
# #     "3_1 100": 180.8906745911,
# # }
# #
# #
# # # extract marabou_after_pp results
# # df_results_marabou_after_pp = pd.read_json("/home/yizhak/Desktop/compare_results_by_property/medium/marabou_after_pp_2019-10-14/outfile_df2json/df_all")
# # df_results_marabou_after_pp_l500 = df_results_marabou_after_pp[df_results_marabou_after_pp.filenames.str.contains("L_500")]
# # df_results_marabou_after_pp_l100 = df_results_marabou_after_pp[df_results_marabou_after_pp.filenames.str.contains("L_100")]
# # for row in df_results_marabou_after_pp_l500.iterrows():
# #     print(row[1]["net name"], float(row[1]["orig_query_time"]))
# # for row in df_results_marabou_after_pp_l100.iterrows():
# #     print(row[1]["net name"], float(row[1]["orig_query_time"]))
# #
# # d_marabou_after_pp = {
# #     # 500
# #     "1_5 500": 362.2670824528,
# #     "1_1 500": 76.3907945156,
# #     "3_9 500": 430.3886027336,
# #     "3_6 500": 105.7820112705,
# #     "5_8 500": 326.1316654682,
# #     "3_8 500": 754.2328572273,
# #     "2_7 500": 82.2208106518,
# #     "1_2 500": 109.851829052,
# #     "3_4 500": 329.9208586216,
# #     "1_6 500": 5.812921524,
# #     "2_1 500": 341.2230713367,
# #     "1_4 500": 76.5244410038,
# #     "4_1 500": 222.4396858215,
# #     "4_7 500": 511.3452959061,
# #     "4_8 500": 697.7116105556,
# #     "4_5 500": 236.3451294899,
# #     "5_1 500": 858.3980236053,
# #     "2_2 500": 1600.8123927116,
# #     "2_3 500": 417.9404368401,
# #     "5_4 500": 320.3789639473,
# #     "5_6 500": 483.5864052773,
# #     "1_7 500": 1.3704514503,
# #     "4_3 500": 472.9419817924,
# #     "4_9 500": 998.454218626,
# #     "2_6 500": 70.6936717033,
# #     "1_9 500": 1.7303872108,
# #     "5_9 500": 1.6433091164,
# #     "5_5 500": 283.4451630116,
# #     "3_3 500": 412.9553496838,
# #     "5_3 500": 538.8637177944,
# #     "3_1 500": 1033.3747372627,
# #     "4_6 500": 438.0562441349,
# #     "1_3 500": 85.016838789,
# #     "2_5 500": 250.8055114746,
# #     "2_9 500": 446.1732587814,
# #     "2_8 500": 865.7779407501,
# #     "3_7 500": 690.8832695484,
# #     "2_4 500": 112.6885688305,
# #     "4_4 500": 268.2437570095,
# #     "4_2 500": 397.2067008018,
# #     "3_2 500": 278.072425127,
# #     "5_7 500": 310.9703567028,
# #     "5_2 500": 418.7479496002,
# #     "1_8 500": 1.6954174042,
# #     "3_5 500": 120.2004857063,
# #     # 100
# #     "2_7 100": 206.24796772,
# #     "1_2 100": 120.1320288181,
# #     "4_6 100": 655.7842912674,
# #     "5_3 100": 8931.2325079441,
# #     "3_6 100": 94.6547648907,
# #     "3_8 100": 594.033413887,
# #     "5_2 100": 402.2007746696,
# #     "5_4 100": 901.7484750748,
# #     "5_6 100": 544.8459291458,
# #     "3_3 100": 543.2201247215,
# #     "5_9 100": 216.8584280014,
# #     "3_1 100": 876.5199193954,
# #     "5_5 100": 341.9165871143,
# #     "4_7 100": 9167.4051060677,
# #     "2_9 100": 631.4636089802,
# #     "3_7 100": 771.4029269218,
# #     "1_1 100": 4752.0783138275,
# #     "4_4 100": 350.1080677509,
# #     "2_4 100": 432.1842739582,
# #     "4_2 100": 449.7442634106,
# #     "3_4 100": 433.7385745049,
# #     "3_9 100": 501.2704999447,
# #     "1_8 100": 101.2362279892,
# #     "2_5 100": 282.8163900375,
# #     "1_9 100": 86.4092702866,
# #     "1_6 100": 245.0721285343,
# #     "4_1 100": 1713.9059414864,
# #     "2_8 100": 186.7788701057,
# #     "1_7 100": 93.9637942314,
# #     "2_1 100": 515.1199755669,
# #     "3_2 100": 291.8588135242,
# #     "3_5 100": 238.3424656391,
# #     "2_6 100": 135.196772337,
# #     "4_3 100": 344.0062541962,
# #     "4_5 100": 351.2085969448,
# #     "5_7 100": 435.2124838829,
# #     "2_3 100": 390.2684519291,
# #     "5_8 100": 280.2288732529,
# #     "4_8 100": 618.4159924984,
# #     "1_5 100": 36.7457959652,
# #     "4_9 100": 26032.785118103,
# #     "1_3 100": 176.2297585011,
# #     "1_4 100": 86.4091215134,
# #     "5_1 100": 800.3053460121,
# # }
#
# # extract best cegarabou results
# df_results_cegarabou = pd.read_json("/home/yizhak/Desktop/compare_results_by_property/long/cegarabou_2019-10-06/outfile_df2json/df_all")
# df_results_cegarabou_complete = df_results_cegarabou[df_results_cegarabou.filenames.str.contains("_A_complete_")]
# df_results_cegarabou_heuristic = df_results_cegarabou[df_results_cegarabou.filenames.str.contains("_A_heuristic_")]
# print(df_results_cegarabou_complete.shape[0])
# print(df_results_cegarabou_heuristic.shape[0])
#
# # complete.shape[0] > heuristic.shape[0] so use it
# df_results_cegarabou_complete_cegar = df_results_cegarabou_complete[df_results_cegarabou_complete.filenames.str.contains("_R_cegar_")]
# df_results_cegarabou_complete_cetar = df_results_cegarabou_complete[df_results_cegarabou_complete.filenames.str.contains("_R_cetar_")]
# print(df_results_cegarabou_complete_cegar.shape[0])
# print(df_results_cegarabou_complete_cetar.shape[0])
#
# # cegar.shape[0] > cetar.shape[0] so use it
# df_results_cegarabou_complete_cegar_as250 = df_results_cegarabou_complete_cegar[df_results_cegarabou_complete_cegar.filenames.str.contains("_AS_250_")]
# df_results_cegarabou_complete_cegar_as100 = df_results_cegarabou_complete_cegar[df_results_cegarabou_complete_cegar.filenames.str.contains("_AS_100_")]
# print(df_results_cegarabou_complete_cegar_as100.shape)
# print(df_results_cegarabou_complete_cegar_as250.shape)
#
# # as100.shape[0] > as250.shape[0] so use it
# df_results_cegarabou_complete_cegar_as100_rs1 = df_results_cegarabou_complete_cegar_as100[df_results_cegarabou_complete_cegar_as100.filenames.str.contains("_RS_1_")]
# df_results_cegarabou_complete_cegar_as100_rs10 = df_results_cegarabou_complete_cegar_as100[df_results_cegarabou_complete_cegar_as100.filenames.str.contains("_RS_10_")]
# df_results_cegarabou_complete_cegar_as100_rs50 = df_results_cegarabou_complete_cegar_as100[df_results_cegarabou_complete_cegar_as100.filenames.str.contains("_RS_50_")]
# df_results_cegarabou_complete_cegar_as100_rs100 = df_results_cegarabou_complete_cegar_as100[df_results_cegarabou_complete_cegar_as100.filenames.str.contains("_RS_100_")]
# print(df_results_cegarabou_complete_cegar_as100_rs1.shape)
# print(df_results_cegarabou_complete_cegar_as100_rs10.shape)
# print(df_results_cegarabou_complete_cegar_as100_rs50.shape)
# print(df_results_cegarabou_complete_cegar_as100_rs100.shape)
#
# # rs50.shape[0] > rs100.shape[0] so use it
# df_results_cegarabou_complete_cegar_as100_rs50_l500 = df_results_cegarabou_complete_cegar_as100_rs50[df_results_cegarabou_complete_cegar_as100_rs50.filenames.str.contains("_L_500_")]
# df_results_cegarabou_complete_cegar_as100_rs50_l100 = df_results_cegarabou_complete_cegar_as100_rs50[df_results_cegarabou_complete_cegar_as100_rs50.filenames.str.contains("_L_100_")]
# print(df_results_cegarabou_complete_cegar_as100_rs50_l500.shape)
# print(df_results_cegarabou_complete_cegar_as100_rs50_l100.shape)
#
# # l500.shape[0] > l100.shape[0], but we use both of them...
#
# df_500 = df_results_cegarabou_complete_cegar_as100_rs50_l500[["net name",  "ar_times"]]
# df_100 = df_results_cegarabou_complete_cegar_as100_rs50_l100[["net name",  "ar_times"]]
#
# # sum ar times
# for row in df_500.iterrows():
#     print(row[1]["net name"], sum(float(x) for x in row[1]["ar_times"][1:-1].split(", ")))
# for row in df_100.iterrows():
#     print(row[1]["net name"], sum(float(x) for x in row[1]["ar_times"][1:-1].split(", ")))
#
# d_cegarabou_sum_ar_times = {
#     # 500
#     "1_3 500": 3.4986047744750977,
#     "4_6 500": 2320.29195189476,
#     "4_2 500": 15.935317516326904,
#     "3_9 500": 387.1654222011566,
#     "2_7 500": 4.375065088272095,
#     "4_1 500": 1257.627371788025,
#     "5_7 500": 1.4681792259216309,
#     "5_3 500": 103.25609755516052,
#     "2_2 500": 588.5378396511078,
#     "5_1 500": 302.2282636165619,
#     "5_9 500": 29.727933406829834,
#     "3_6 500": 419.7206370830536,
#     "4_7 500": 5.286901473999023,
#     "1_9 500": 225.91684675216675,
#     "4_3 500": 221.46340823173523,
#     "5_6 500": 2.4357283115386963,
#     "1_5 500": 925.1434228420258,
#     "5_8 500": 16.670150995254517,
#     "4_5 500": 6.230702877044678,
#     "2_9 500": 4.9583470821380615,
#     "1_8 500": 76.79796290397644,
#     "3_3 500": 147.42733907699585,
#     "5_2 500": 183.0510048866272,
#     "3_5 500": 0.8696150779724121,
#     "2_1 500": 209.9809181690216,
#     "3_8 500": 1.403045654296875,
#     "5_4 500": 469.51030468940735,
#     "4_8 500": 5.539355516433716,
#     "1_7 500": 544.3518562316895,
#     "3_2 500": 1489.683235168457,
#     "4_4 500": 313.4217686653137,
#     "3_7 500": 10.22359585762024,
#     "1_4 500": 350.3835401535034,
#     "4_9 500": 1.251405954360962,
#     "1_1 500": 5.2782087326049805,
#     "1_6 500": 1960.9676156044006,
#     "3_4 500": 543.9217445850372,
#     "2_8 500": 1.3097984790802002,
#     "1_2 500": 496.410076379776,
#     # 100
#     "3_4 100": 943.7079570293427,
#     "4_8 100": 40.83619213104248,
#     "2_2 100": 27349.997271299362,
#     "5_4 100": 851.8604083061218,
#     "3_8 100": 4.861781358718872,
#     "3_7 100": 21.497034311294556,
#     "3_3 100": 1172.6140773296356,
#     "4_1 100": 2128.723603963852,
#     "1_8 100": 102.1031105518341,
#     "4_2 100": 65.85559225082397,
#     "2_7 100": 32.93417429924011,
#     "5_3 100": 483.115473985672,
#     "5_1 100": 551.9554200172424,
#     "3_5 100": 16.942864179611206,
#     "5_6 100": 1.596494436264038,
#     "1_3 100": 0.9343054294586182,
#     "3_6 100": 1211.830361366272,
#     "3_9 100": 1834.2880971431732,
#     "4_5 100": 1.1054131984710693,
#     "5_7 100": 3.212557792663574,
#     "1_5 100": 180.12356519699097,
#     "4_4 100": 335.81499314308167,
#     "2_1 100": 2902.3595354557037,
#     "5_8 100": 4.997140645980835,
#     "2_8 100": 14.935363531112671,
#     "4_3 100": 2611.2765543460846,
#     "4_7 100": 1.6906914710998535,
#     "4_6 100": 1139.8554735183716,
#     "1_4 100": 3201.0072572231293,
#     "5_2 100": 539.5132234096527,
#     "1_9 100": 139.01236486434937,
#     "5_9 100": 4.359238624572754,
#     "2_9 100": 54.99050760269165,
#     "3_2 100": 6202.876989364624,
#     "1_6 100": 2674.69619345665,
#     "1_1 100": 3.830270290374756,
#     "1_2 100": 773.754650592804,
#     "4_9 100": 9.936185121536255,
#     "1_7 100": 2318.324691295624,
# }
#
#
# # last query time
# for row in df_500.iterrows():
#     print(row[1]["net name"], float(row[1]["ar_times"][1:-1].split(", ")[-1]))
# for row in df_100.iterrows():
#     print(row[1]["net name"], float(row[1]["ar_times"][1:-1].split(", ")[-1]))
#
# d_cegarabou_last_query_time = {
#     # 500
#     "1_3 500": 3.4986047744750977,
#     "4_6 500": 474.1722173690796,
#     "4_2 500": 14.497947454452515,
#     "3_9 500": 49.92515850067139,
#     "2_7 500": 4.375065088272095,
#     "4_1 500": 202.23075103759766,
#     "5_7 500": 1.4681792259216309,
#     "5_3 500": 27.035215139389038,
#     "2_2 500": 192.79024076461792,
#     "5_1 500": 64.78479242324829,
#     "5_9 500": 29.727933406829834,
#     "3_6 500": 94.5000388622284,
#     "4_7 500": 5.286901473999023,
#     "1_9 500": 8.738630771636963,
#     "4_3 500": 45.48619222640991,
#     "5_6 500": 2.4357283115386963,
#     "1_5 500": 193.22966384887695,
#     "5_8 500": 16.670150995254517,
#     "4_5 500": 6.230702877044678,
#     "2_9 500": 4.9583470821380615,
#     "1_8 500": 18.237333297729492,
#     "3_3 500": 40.99721574783325,
#     "5_2 500": 52.861698627471924,
#     "3_5 500": 0.8696150779724121,
#     "2_1 500": 51.801865100860596,
#     "3_8 500": 1.403045654296875,
#     "5_4 500": 75.48629450798035,
#     "4_8 500": 5.539355516433716,
#     "1_7 500": 96.34817576408386,
#     "3_2 500": 378.8729703426361,
#     "4_4 500": 76.25922441482544,
#     "3_7 500": 10.22359585762024,
#     "1_4 500": 96.16750741004944,
#     "4_9 500": 1.251405954360962,
#     "1_1 500": 5.2782087326049805,
#     "1_6 500": 239.5860915184021,
#     "3_4 500": 79.98718619346619,
#     "2_8 500": 1.3097984790802002,
#     "1_2 500": 57.31457877159119,
#     # 100
#     "3_4 100": 189.5051109790802,
#     "4_8 100": 40.83619213104248,
#     "2_2 100": 26636.01070666313,
#     "5_4 100": 166.58965253829956,
#     "3_8 100": 4.861781358718872,
#     "3_7 100": 21.497034311294556,
#     "3_3 100": 292.3159718513489,
#     "4_1 100": 442.117556810379,
#     "1_8 100": 27.425087690353394,
#     "4_2 100": 44.37616205215454,
#     "2_7 100": 32.93417429924011,
#     "5_3 100": 183.3163993358612,
#     "5_1 100": 131.65136861801147,
#     "3_5 100": 16.942864179611206,
#     "5_6 100": 1.596494436264038,
#     "1_3 100": 0.9343054294586182,
#     "3_6 100": 388.87661504745483,
#     "3_9 100": 316.7972447872162,
#     "4_5 100": 1.1054131984710693,
#     "5_7 100": 3.212557792663574,
#     "1_5 100": 53.84671711921692,
#     "4_4 100": 54.93575954437256,
#     "2_1 100": 604.6206107139587,
#     "5_8 100": 4.997140645980835,
#     "2_8 100": 14.935363531112671,
#     "4_3 100": 736.9747352600098,
#     "4_7 100": 1.6906914710998535,
#     "4_6 100": 296.62767148017883,
#     "1_4 100": 764.1890618801117,
#     "5_2 100": 120.96468615531921,
#     "1_9 100": 37.165825843811035,
#     "5_9 100": 4.359238624572754,
#     "2_9 100": 54.99050760269165,
#     "3_2 100": 3041.8806467056274,
#     "1_6 100": 515.5869834423065,
#     "1_1 100": 3.830270290374756,
#     "1_2 100": 186.28123545646667,
#     "4_9 100": 9.936185121536255,
#     "1_7 100": 208.51937675476074,
# }
#
#
# d_marabou_before_pp_l500 = {k: v for (k,v) in d_marabou_before_pp.items() if " 500" in k}
# d_marabou_before_pp_l100 = {k: v for (k,v) in d_marabou_before_pp.items() if " 100" in k}
#
# #results_long_marabou_after_pp = {"1_8 500": 4.3, "1_9 500": 1.3}
# d_marabou_after_pp_l500 = {k: v for (k,v) in d_marabou_after_pp.items() if " 500" in k}
# d_marabou_after_pp_l100 = {k: v for (k,v) in d_marabou_after_pp.items() if " 100" in k}
#
#
# for title, d_cegarabou in [("d_cegarabou_last_query_time", d_cegarabou_last_query_time),
#                                     ("d_cegarabou_sum_ar_times", d_cegarabou_sum_ar_times)]:
#     print("-" * 80)
#     print(title)
#     print("-" * 80)
#
#     # cegarabo_vs_marabou_before_pp
#     print("-" * 40)
#     print("cegarabo_vs_marabou_before_pp")
#     print("-" * 40)
#     cegarabo_vs_marabou_before_pp = []
#     TIMEOUT_VAL = 100000
#     for k,v in d_cegarabou.items():
#         if k in d_marabou_before_pp.keys():
#             item = (v, d_marabou_before_pp[k])
#         else:
#             item = (v, TIMEOUT_VAL)
#         cegarabo_vs_marabou_before_pp.append(item)
#     for k, v in d_marabou_before_pp.items():
#         if k in d_cegarabou.keys():
#             continue  # the item is already in the list
#         else:
#             item = (TIMEOUT_VAL, v)
#             cegarabo_vs_marabou_before_pp.append(item)
#
#     finishes = [(x,y) for (x,y) in cegarabo_vs_marabou_before_pp if x != TIMEOUT_VAL and y != TIMEOUT_VAL]
#     timeouts = [(x,y) for (x,y) in cegarabo_vs_marabou_before_pp if x == TIMEOUT_VAL or y == TIMEOUT_VAL]
#     x_finishes = [x for (x,y) in finishes]
#     y_finishes = [y for (x,y) in finishes]
#     x_timeouts = [x for (x,y) in timeouts]
#     y_timeouts = [y for (x,y) in timeouts]
#
#     finished_points = plt.scatter(x=x_finishes, y=y_finishes, color="b", marker="o")
#     timeout_points = plt.scatter(x=x_timeouts, y=y_timeouts, color="r", marker="x")
#     # plt.scatter(x=cegarabou_results, y=marabou_before_pp_results)
#     # timeouts lines
#     vertical_timeout_line = plt.axvline(x=TIMEOUT_VAL, ymin=0, ymax=TIMEOUT_VAL, color='c', linestyle='--')
#     horizontal_timeout_line = plt.axhline(y=TIMEOUT_VAL, xmin=0, xmax=TIMEOUT_VAL, color='c', linestyle='--')
#     # y=x line
#     y_equals_x_line = plt.plot(list(range(0, int(TIMEOUT_VAL))), 'g--', label="y=x")
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel("Marabou with Abstraction")
#     plt.ylabel("Marabou")
#     plt.legend(
#         (vertical_timeout_line, y_equals_x_line, timeout_points, finished_points),
#         ("timeout line", "y=x", "timeout experiment", "finished experiment"),
#         scatterpoints=1,
#         ncol=1,
#         fontsize=8
#     )
#     plt.show()
#
#
#
#
#
#
#
#
#
#
#     # cegarabo_vs_marabou_after_pp
#     print("-" * 40)
#     print("cegarabo_vs_marabou_after_pp")
#     print("-" * 40)
#     cegarabo_vs_marabou_after_pp = []
#     TIMEOUT_VAL = 100000
#     for k,v in d_cegarabou.items():
#         if k in d_marabou_after_pp.keys():
#             item = (v, d_marabou_after_pp[k])
#         else:
#             item = (v, TIMEOUT_VAL)
#         cegarabo_vs_marabou_after_pp.append(item)
#     for k, v in d_marabou_after_pp.items():
#         if k in d_cegarabou.keys():
#             continue  # the item is already in the list
#         else:
#             item = (TIMEOUT_VAL, v)
#             cegarabo_vs_marabou_after_pp.append(item)
#
#     finishes = [(x,y) for (x,y) in cegarabo_vs_marabou_after_pp if x != TIMEOUT_VAL and y != TIMEOUT_VAL]
#     timeouts = [(x,y) for (x,y) in cegarabo_vs_marabou_after_pp if x == TIMEOUT_VAL or y == TIMEOUT_VAL]
#     x_finishes = [x for (x,y) in finishes]
#     y_finishes = [y for (x,y) in finishes]
#     x_timeouts = [x for (x,y) in timeouts]
#     y_timeouts = [y for (x,y) in timeouts]
#
#     finished_points = plt.scatter(x=x_finishes, y=y_finishes, color="b", marker="o")
#     timeout_points = plt.scatter(x=x_timeouts, y=y_timeouts, color="r", marker="x")
#     # plt.scatter(x=cegarabou_results, y=marabou_after_pp_results)
#     # timeouts lines
#     vertical_timeout_line = plt.axvline(x=TIMEOUT_VAL, ymin=0, ymax=TIMEOUT_VAL, color='c', linestyle='--')
#     horizontal_timeout_line = plt.axhline(y=TIMEOUT_VAL, xmin=0, xmax=TIMEOUT_VAL, color='c', linestyle='--')
#     # y=x line
#     y_equals_x_line = plt.plot(list(range(0, int(TIMEOUT_VAL))), 'g--', label="y=x")
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel("Marabou with Abstraction")
#     plt.ylabel("Marabou")
#     plt.legend(
#         (vertical_timeout_line, y_equals_x_line, timeout_points, finished_points),
#         ("timeout line", "y=x", "timeout experiment", "finished experiment"),
#         scatterpoints=1,
#         ncol=1,
#         fontsize=8
#     )
#     plt.show()
