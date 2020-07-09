import os
import random
import datetime
import glob
import pathlib


def create_content(identities_list, identity, cameras, place):
    if place is None:
        place = [str(random.randrange(1, 4)).zfill(2),
                 str(random.randrange(1, 4)).zfill(2),
                 str(random.randrange(1, 4)).zfill(2)]
    else:
        place = [str(place).zfill(2)]*3

    first_floor = glob.glob(os.getenv('VIDEOS_ROUTE') + "/averobot_floor_01/" + str(identities_list[identity]).zfill(2)
                            + "_01_" + place[0] + "_*_" + str(cameras[0]).zfill(2)
                            + ".mp4")

    second_floor = glob.glob(os.getenv('VIDEOS_ROUTE') + "/averobot_floor_02/" + str(identities_list[identity]).zfill(2)
                             + "_02_" + place[1] + "_*_" + str(cameras[1]).zfill(2)
                             + ".mp4")

    third_floor = glob.glob(os.getenv('VIDEOS_ROUTE') + "/averobot_floor_03/" + str(identities_list[identity]).zfill(2)
                            + "_03_" + place[2] + "_*_" + str(cameras[2]).zfill(2)
                            + ".mp4")

    return [os.path.join(*pathlib.Path(first_floor[0]).parts[-2:]).replace("\\", "/") + "\n",
            os.path.join(*pathlib.Path(second_floor[0]).parts[-2:]).replace("\\", "/") + "\n",
            os.path.join(*pathlib.Path(third_floor[0]).parts[-2:]).replace("\\", "/") + "\n"]


def create_indexer_files(files_pairs=1, training_identities_number=55, test_identities_number=56, cameras=(2, 3, 8),
                         place=None):
    assert (training_identities_number + test_identities_number <= int(os.getenv('IDENTITIES_NUMBER'))), \
        "Training Identities + Test Identities cannot be > " + os.getenv('IDENTITIES_NUMBER')

    assert (cameras[0] == 1 or cameras[0] == 2), "Cameras for 1st floor must be '1' or '2'"
    assert (cameras[1] >= 3 or cameras[1] <= 5), "Cameras for 2nd floor must be '3', '4' or '5'"
    assert (cameras[2] >= 6 or cameras[2] <= 8), "Cameras for 3rd floor must be '6', '7' or '8'"
    assert (place is None or 1 <= place <= 3), "Place must be 'None' for random place or '1', '2' or '3' " \
                                               "(stairs, lift or corridor)"

    print("[Creating indexer files... (" + str(files_pairs) + " files)]")
    for _ in range(files_pairs):
        identities_list = list(range(1, int(os.getenv('IDENTITIES_NUMBER')) + 1))
        random.shuffle(identities_list)

        file_name = os.getenv('MAIN_ROUTE') + "/out/file-indexer/" + datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
        train_file = open(file_name + "-train.txt", "a")
        test_file = open(file_name + "-test.txt", "a")

        for training_identity in range(0, training_identities_number):
            first_floor, second_floor, third_floor = create_content(identities_list, training_identity, cameras, place)
            train_file.writelines([first_floor, second_floor, third_floor])

        for test_identity in range(training_identities_number, training_identities_number + test_identities_number):
            first_floor, second_floor, third_floor = create_content(identities_list, test_identity, cameras, place)
            test_file.writelines([first_floor, second_floor, third_floor])

        train_file.close()
        test_file.close()

    print("[Indexer files made (" + str(files_pairs) + " file(s) pair(s))]")
