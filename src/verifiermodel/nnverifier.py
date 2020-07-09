import itertools
import random
import os
import numpy
import datetime
import pandas as pd

from sklearn.metrics.pairwise import euclidean_distances
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Input, concatenate


class NNVerifier:
    def __init__(self, train_videos_embeddings_1, test_videos_embeddings_1,
                 train_videos_embeddings_2=None, test_videos_embeddings_2=None,
                 mlp_type='default', select_one_frame=False, use_one_detector=False,
                 use_kolmogorov_theorem=False, use_euclidean_distance=False):
        if not use_one_detector:
            assert (numpy.array_equal(sorted(train_videos_embeddings_1['video_id'].unique()),
                                      sorted(train_videos_embeddings_2['video_id'].unique()))), \
                "Training videos embeddings must have same videos"

            assert (numpy.array_equal(sorted(test_videos_embeddings_1['video_id'].unique()),
                                      sorted(test_videos_embeddings_2['video_id'].unique()))), \
                "Test videos embeddings must have same videos"
            self.train_videos_embeddings_2 = train_videos_embeddings_2
            self.test_videos_embeddings_2 = test_videos_embeddings_2
            self.videos_embeddings_2 = pd.concat([train_videos_embeddings_2, test_videos_embeddings_2])

        assert (mlp_type == 'default' or mlp_type == 'subtract' or mlp_type == 'abs_subtract'
                or (mlp_type == 'abs_subtract_keep' and use_one_detector is True)),\
                "Error in 'mlp_type' parameter. Please, use one of the following values: " \
                "'default', 'subtract', 'abs_subtract' or" "abs_subtract_keep"

        self.train_videos_embeddings_1 = train_videos_embeddings_1
        self.test_videos_embeddings_1 = test_videos_embeddings_1
        self.videos_embeddings_1 = pd.concat([train_videos_embeddings_1, test_videos_embeddings_1])
        self.mlp_type = mlp_type
        self.select_one_frame = select_one_frame
        self.use_one_detector = use_one_detector
        self.use_kolmogorov_theorem = use_kolmogorov_theorem
        self.use_euclidean_distance = use_euclidean_distance

    def get_embeddings_pairs(self, pair_type='train'):

        assert (pair_type == 'train' or pair_type == 'test'), "Error in 'pair_type' parameter. Please, use one of the" \
                                                              "following values: 'train' or 'test'."

        same_person_embeddings_pairs = []
        different_people_embeddings_pairs = []

        if self.use_euclidean_distance:
            same_person_ed = []
            different_people_ed = []

        if pair_type == 'train':
            videos_ids = self.train_videos_embeddings_1['video_id'].unique()
        elif pair_type == 'test':
            videos_ids = self.test_videos_embeddings_1['video_id'].unique()

        print("[Getting all possible pairs...]")

        for index, video_id in enumerate(videos_ids):
            print("[Progress: " + str(index) + "/" + str(len(videos_ids)) + "]", end='\r')

            if pair_type == 'train':
                person_videos_embeddings_1 = self.train_videos_embeddings_1[self.train_videos_embeddings_1['video_id'] == video_id]
                if not self.use_one_detector:
                    person_videos_embeddings_2 = self.train_videos_embeddings_2[self.train_videos_embeddings_2['video_id'] == video_id]
            elif pair_type == 'test':
                person_videos_embeddings_1 = self.test_videos_embeddings_1[self.test_videos_embeddings_1['video_id'] == video_id]
                if not self.use_one_detector:
                    person_videos_embeddings_2 = self.test_videos_embeddings_2[self.test_videos_embeddings_2['video_id'] == video_id]

            floor = person_videos_embeddings_1['floor'].unique()[0]
            person_id = person_videos_embeddings_1['id'].unique()[0]
            if not self.select_one_frame:
                if not self.use_one_detector:
                    same_video_embeddings_pairs = list(itertools.product(person_videos_embeddings_1.iloc[:, 3:].to_numpy(),
                                                                         person_videos_embeddings_2.iloc[:, 3:].to_numpy()))
                else:
                    same_video_embeddings_pairs = person_videos_embeddings_1.iloc[:, 3:].to_numpy()
            else:
                if not self.use_one_detector:
                    same_video_embeddings_pairs = (person_videos_embeddings_1.iloc[0, 3:].to_numpy(),
                                                   person_videos_embeddings_2.iloc[0, 3:].to_numpy())
                else:
                    same_video_embeddings_pairs = person_videos_embeddings_1.iloc[0, 3:].to_numpy()



            for different_video in list(set(videos_ids) - {video_id}):
                if pair_type == 'train':
                    person_videos_embeddings_3 = self.train_videos_embeddings_1[self.train_videos_embeddings_1['video_id'] == different_video]
                    if not self.use_one_detector:
                        person_videos_embeddings_4 = self.train_videos_embeddings_2[self.train_videos_embeddings_2['video_id'] == different_video]
                elif pair_type == 'test':
                    person_videos_embeddings_3 = self.test_videos_embeddings_1[self.test_videos_embeddings_1['video_id'] == different_video]
                    if not self.use_one_detector:
                        person_videos_embeddings_4 = self.test_videos_embeddings_2[self.test_videos_embeddings_2['video_id'] == different_video]

                different_video_floor = person_videos_embeddings_3['floor'].unique()[0]
                if floor == different_video_floor:
                    continue
                different_video_person_id = person_videos_embeddings_3['id'].unique()[0]

                if not self.select_one_frame:
                    if not self.use_one_detector:
                        different_video_embeddings_pairs = list(itertools.product(person_videos_embeddings_3.iloc[:, 3:].to_numpy(),
                                                                                  person_videos_embeddings_4.iloc[:, 3:].to_numpy()))
                    else:
                        different_video_embeddings_pairs = person_videos_embeddings_3.iloc[:, 3:].to_numpy()

                    embeddings_pairs = list(itertools.product(same_video_embeddings_pairs,
                                                              different_video_embeddings_pairs))
                else:
                    if not self.use_one_detector:
                        different_video_embeddings_pairs = (person_videos_embeddings_3.iloc[0, 3:].to_numpy(),
                                                            person_videos_embeddings_4.iloc[0, 3:].to_numpy())
                    else:
                        different_video_embeddings_pairs = person_videos_embeddings_3.iloc[0, 3:].to_numpy()

                    embeddings_pairs = [(same_video_embeddings_pairs,
                                         different_video_embeddings_pairs)]

                if self.use_euclidean_distance:
                    ed_list = list()
                    for embedding_pair_index in range(len(embeddings_pairs)):
                        if not self.use_one_detector:
                            ed_list.append(euclidean_distances([numpy.concatenate((embeddings_pairs[embedding_pair_index][0][0],
                                                                                   embeddings_pairs[embedding_pair_index][0][1]))],
                                                               [numpy.concatenate((embeddings_pairs[embedding_pair_index][1][0],
                                                                                   embeddings_pairs[embedding_pair_index][1][1]))])[0])
                        else:
                            ed_list.append(euclidean_distances([embeddings_pairs[embedding_pair_index][0]],
                                                               [embeddings_pairs[embedding_pair_index][1]])[0])

                if self.mlp_type == 'subtract' or self.mlp_type == 'abs_subtract' or self.mlp_type == 'abs_subtract_keep':
                    for embedding_pair_index in range(len(embeddings_pairs)):
                        if not self.use_one_detector:
                            embeddings_pairs[embedding_pair_index] = numpy.subtract(numpy.concatenate((embeddings_pairs[embedding_pair_index][0][0],
                                                                                                       embeddings_pairs[embedding_pair_index][0][1])),
                                                                                    numpy.concatenate((embeddings_pairs[embedding_pair_index][1][0],
                                                                                                       embeddings_pairs[embedding_pair_index][1][1])))
                        else:
                            if self.mlp_type == 'abs_subtract_keep':
                                embeddings_pairs[embedding_pair_index] = numpy.concatenate((embeddings_pairs[embedding_pair_index][0],
                                                                                            embeddings_pairs[embedding_pair_index][1],
                                                                                            numpy.absolute(numpy.subtract(embeddings_pairs[embedding_pair_index][0],
                                                                                                                          embeddings_pairs[embedding_pair_index][1]))))
                            else:
                                embeddings_pairs[embedding_pair_index] = numpy.subtract(embeddings_pairs[embedding_pair_index][0],
                                                                                        embeddings_pairs[embedding_pair_index][1])
                        if self.mlp_type == 'abs_subtract':
                            embeddings_pairs[embedding_pair_index] = numpy.absolute(embeddings_pairs[embedding_pair_index])

                if person_id == different_video_person_id:
                    same_person_embeddings_pairs.extend(embeddings_pairs)
                    if self.use_euclidean_distance:
                        same_person_ed.extend(ed_list)
                else:
                    different_people_embeddings_pairs.extend(embeddings_pairs)
                    if self.use_euclidean_distance:
                        different_people_ed.extend(ed_list)

        print("[All pairs have been generated successfully (" + str(len(same_person_embeddings_pairs)) +
              " same person pairs & " + str(len(different_people_embeddings_pairs)) + " different people pairs)")

        if self.use_euclidean_distance:
            return [same_person_embeddings_pairs, different_people_embeddings_pairs,
                    same_person_ed, different_people_ed]

        return [same_person_embeddings_pairs, different_people_embeddings_pairs]

    def create_mlp(self, print_summary=True):
        if self.use_one_detector and self.mlp_type == 'default':
            input_dim = 256
        elif self.use_one_detector and self.mlp_type == 'abs_subtract_keep':
            input_dim = 384
        elif self.use_one_detector and (self.mlp_type == 'subtract' or self.mlp_type == 'abs_subtract'):
            input_dim = 128
        elif self.mlp_type == 'default':
            input_dim = 512
        else:
            input_dim = 256
        if not self.use_euclidean_distance:
            self.model = Sequential()
            if self.use_kolmogorov_theorem:
                self.model.add(Dense(input_dim, input_dim=input_dim, activation='relu'))
                self.model.add(Dense((2*input_dim)+1, activation='relu'))
                self.model.add(Dense(1, activation='sigmoid'))
            else:
                self.model.add(Dense(2048, input_dim=input_dim, activation='relu'))
                self.model.add(Dropout(0.5))
                self.model.add(Dense(512, activation='relu'))
                self.model.add(Dropout(0.5))
                self.model.add(Dense(64, activation='relu'))
                self.model.add(Dropout(0.25))
                self.model.add(Dense(1, activation='sigmoid'))
        else:
            ed_input = Input(shape=[1], name='euclidean_distance')

            input_layer = Input(shape=[input_dim])
            # hidden_layer_1 = Dense(input_dim, activation='relu')(input_layer)
            # hidden_layer_1_ed = concatenate([hidden_layer_1, ed_input])
            # hidden_layer_2 = Dense((input_dim*2)+1, activation='relu')(hidden_layer_1_ed)
            # output_layer = Dense(1, activation='sigmoid')(hidden_layer_2)

            hidden_layer_1 = Dense(input_dim, activation='relu')(input_layer)
            hidden_layer_2 = Dense(int(input_dim/2), activation='relu')(hidden_layer_1)
            hidden_layer_2_ed = concatenate([hidden_layer_2, ed_input])
            output_layer = Dense(1, activation='sigmoid')(hidden_layer_2_ed)

            self.model = Model(inputs=[input_layer, ed_input], outputs=output_layer)

        if print_summary:
            self.model.summary()

    def train_model(self, same_person_embeddings_pairs_train, different_people_embeddings_pairs_train,
                    batch_size=50000, epochs=15, same_person_ed=None, different_people_ed=None):

        same_person_embeddings_pairs_train = numpy.array(same_person_embeddings_pairs_train)

        if self.use_euclidean_distance:
            shuffle_list = list(range(0, len(same_person_embeddings_pairs_train)))
            random.shuffle(shuffle_list)
            different_people_embeddings_pairs_train = numpy.array(different_people_embeddings_pairs_train)[shuffle_list]
            different_people_ed = numpy.array(different_people_ed)[shuffle_list]
            ed_train = numpy.concatenate((same_person_ed, different_people_ed))
        else:
            random.shuffle(different_people_embeddings_pairs_train)
            different_people_embeddings_pairs_train = numpy.array(different_people_embeddings_pairs_train[:same_person_embeddings_pairs_train.shape[0]])

        if self.use_one_detector and self.mlp_type == 'default':
            same_person_embeddings_pairs_train = same_person_embeddings_pairs_train.reshape((same_person_embeddings_pairs_train.shape[0],
                                                                                             256))
            different_people_embeddings_pairs_train = different_people_embeddings_pairs_train.reshape((different_people_embeddings_pairs_train.shape[0],
                                                                                                       256))
        elif self.mlp_type == 'default':
            same_person_embeddings_pairs_train = same_person_embeddings_pairs_train.reshape((same_person_embeddings_pairs_train.shape[0],
                                                                                             512))
            different_people_embeddings_pairs_train = different_people_embeddings_pairs_train.reshape((different_people_embeddings_pairs_train.shape[0],
                                                                                                       512))

        X_train = numpy.concatenate((same_person_embeddings_pairs_train,
                                     different_people_embeddings_pairs_train))

        Y_train = numpy.concatenate((numpy.zeros((same_person_embeddings_pairs_train.shape[0], 1)),
                                     numpy.ones((different_people_embeddings_pairs_train.shape[0], 1))))

        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        shuffle_list = list(range(0, X_train.shape[0]))
        random.shuffle(shuffle_list)
        X_train = X_train[shuffle_list]
        Y_train = Y_train[shuffle_list]

        if self.use_euclidean_distance:
            ed_train = ed_train[shuffle_list]
            self.history = self.model.fit([X_train, ed_train], Y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                                          validation_split=0.2)
        else:
            self.history = self.model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                                          validation_split=0.2)

    def evaluate(self, same_person_embeddings_pairs_test, different_people_embeddings_pairs_test,
                 same_person_ed=None, different_people_ed=None):

        random.shuffle(different_people_embeddings_pairs_test)

        same_person_embeddings_pairs_test = numpy.array(same_person_embeddings_pairs_test)

        if self.use_euclidean_distance:
            shuffle_list = list(range(0, len(same_person_embeddings_pairs_test)))
            random.shuffle(shuffle_list)
            different_people_embeddings_pairs_test = numpy.array(different_people_embeddings_pairs_test)[shuffle_list]
            different_people_ed = numpy.array(different_people_ed)[shuffle_list]
            ed_train = numpy.concatenate((same_person_ed, different_people_ed))
        else:
            random.shuffle(different_people_embeddings_pairs_test)
            different_people_embeddings_pairs_test = numpy.array(different_people_embeddings_pairs_test[:same_person_embeddings_pairs_test.shape[0]])

        if self.use_one_detector and self.mlp_type == 'default':
            same_person_embeddings_pairs_test = same_person_embeddings_pairs_test.reshape((same_person_embeddings_pairs_test.shape[0],
                                                                                           256))
            different_people_embeddings_pairs_test = different_people_embeddings_pairs_test.reshape((different_people_embeddings_pairs_test.shape[0],
                                                                                                     256))
        elif self.mlp_type == 'default':
            same_person_embeddings_pairs_test = same_person_embeddings_pairs_test.reshape((same_person_embeddings_pairs_test.shape[0],
                                                                                           512))
            different_people_embeddings_pairs_test = different_people_embeddings_pairs_test.reshape((different_people_embeddings_pairs_test.shape[0],
                                                                                                     512))

        X_test = numpy.concatenate((same_person_embeddings_pairs_test,
                                    different_people_embeddings_pairs_test))
        Y_test = numpy.concatenate((numpy.zeros((same_person_embeddings_pairs_test.shape[0], 1)),
                                    numpy.ones((different_people_embeddings_pairs_test.shape[0], 1))))

        shuffle_list = list(range(0, X_test.shape[0]))
        random.shuffle(shuffle_list)
        X_test = X_test[shuffle_list]
        Y_test = Y_test[shuffle_list]

        if self.use_euclidean_distance:
            ed_train = ed_train[shuffle_list]
            evaluation_results = self.model.evaluate([X_test, ed_train], Y_test, verbose=1)
        else:
            evaluation_results = self.model.evaluate(X_test, Y_test, verbose=1)

        print("Accuracy: " + str(round(evaluation_results[1], 3)) + " | Loss: " + str(round(evaluation_results[0], 3)))

    def is_same_person(self, id_video_1, id_video_2):
        video_1_detector_1_embeddings = self.videos_embeddings_1[self.videos_embeddings_1['video_id'] == id_video_1]
        video_2_detector_1_embeddings = self.videos_embeddings_1[self.videos_embeddings_1['video_id'] == id_video_2]
        if not self.use_one_detector:
            video_1_detector_2_embeddings = self.videos_embeddings_2[self.videos_embeddings_2['video_id'] == id_video_1]
            video_2_detector_2_embeddings = self.videos_embeddings_2[self.videos_embeddings_2['video_id'] == id_video_2]

        if not self.select_one_frame:
            if not self.use_one_detector:
                video_1_embeddings = list(itertools.product(video_1_detector_1_embeddings.iloc[:, 3:].to_numpy(),
                                                            video_1_detector_2_embeddings.iloc[:, 3:].to_numpy()))

                video_2_embeddings = list(itertools.product(video_2_detector_1_embeddings.iloc[:, 3:].to_numpy(),
                                                            video_2_detector_2_embeddings.iloc[:, 3:].to_numpy()))

            else:
                video_1_embeddings = video_1_detector_1_embeddings.iloc[:, 3:].to_numpy()
                video_2_embeddings = video_2_detector_1_embeddings.iloc[:, 3:].to_numpy()

            nn_vf_input = list(itertools.product(video_1_embeddings,
                                                 video_2_embeddings))
        else:
            if not self.use_one_detector:
                video_1_embeddings = (video_1_detector_1_embeddings.iloc[0, 3:].to_numpy(),
                                      video_1_detector_2_embeddings.iloc[0, 3:].to_numpy())

                video_2_embeddings = (video_2_detector_1_embeddings.iloc[0, 3:].to_numpy(),
                                      video_2_detector_2_embeddings.iloc[0, 3:].to_numpy())
            else:
                video_1_embeddings = video_1_detector_1_embeddings.iloc[0, 3:].to_numpy()
                video_2_embeddings = video_2_detector_1_embeddings.iloc[0, 3:].to_numpy()

            nn_vf_input = [(video_1_embeddings, video_2_embeddings)]

        if self.use_euclidean_distance:
            nn_vf_ed_list = list()
            for embedding_pair_index in range(len(nn_vf_input)):
                if not self.use_one_detector:
                    nn_vf_ed_list.append(euclidean_distances([numpy.concatenate((nn_vf_input[embedding_pair_index][0][0],
                                                                                 nn_vf_input[embedding_pair_index][0][1]))],
                                                             [numpy.concatenate((nn_vf_input[embedding_pair_index][1][0],
                                                                                 nn_vf_input[embedding_pair_index][1][1]))])[0])
                else:
                    nn_vf_ed_list.append(euclidean_distances([nn_vf_input[embedding_pair_index][0]],
                                                             [nn_vf_input[embedding_pair_index][1]])[0])
            nn_vf_ed_list = numpy.array(nn_vf_ed_list)

        if self.mlp_type == 'subtract' or self.mlp_type == 'abs_subtract' or self.mlp_type == 'abs_subtract_keep':
            for embedding_pair_index in range(len(nn_vf_input)):
                if not self.use_one_detector:
                    nn_vf_input[embedding_pair_index] = numpy.subtract(numpy.concatenate((nn_vf_input[embedding_pair_index][0][0],
                                                                                          nn_vf_input[embedding_pair_index][0][1])),
                                                                       numpy.concatenate((nn_vf_input[embedding_pair_index][1][0],
                                                                                          nn_vf_input[embedding_pair_index][1][1])))
                else:
                    if self.mlp_type == 'abs_subtract_keep':
                        nn_vf_input[embedding_pair_index] = numpy.concatenate((nn_vf_input[embedding_pair_index][0],
                                                                               nn_vf_input[embedding_pair_index][1],
                                                                               numpy.absolute(numpy.subtract(nn_vf_input[embedding_pair_index][0],
                                                                                                             nn_vf_input[embedding_pair_index][1]))))
                    else:
                        nn_vf_input[embedding_pair_index] = numpy.subtract(nn_vf_input[embedding_pair_index][0],
                                                                           nn_vf_input[embedding_pair_index][1])
                if self.mlp_type == 'abs_subtract':
                    nn_vf_input[embedding_pair_index] = numpy.absolute(nn_vf_input[embedding_pair_index])

            nn_vf_input = numpy.array(nn_vf_input)
        elif self.mlp_type == 'default':
            nn_vf_input = numpy.array(nn_vf_input)
            if self.use_one_detector:
                nn_vf_input = nn_vf_input.reshape((nn_vf_input.shape[0], 256))
            else:
                nn_vf_input = nn_vf_input.reshape((nn_vf_input.shape[0], 512))

        if self.use_euclidean_distance:
            nn_vf_output = self.model.predict([nn_vf_input, nn_vf_ed_list])
            same_person_samples = numpy.where(nn_vf_output < 0.5)[0].shape[0]
        else:
            nn_vf_output = self.model.predict_classes(nn_vf_input)
            same_person_samples = numpy.where(nn_vf_output == 0)[0].shape[0]

        same_person_rate = same_person_samples / nn_vf_output.shape[0]

        if same_person_rate >= 0.5:
            return {'is_same_person': True,
                    'confidence': round(same_person_rate, 3),
                    'results': nn_vf_output}

        return {'is_same_person': False,
                'confidence': round(1 - same_person_rate, 3),
                'results': nn_vf_output}

    def save_model(self, model_name):
        self.model.save(os.getenv('MAIN_ROUTE') + "/out/verifier-model/" +
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f-") + model_name + '.h5')

    def load_model(self, model_name):
        self.model = load_model(os.getenv('MAIN_ROUTE') + "/out/verifier-model/" + model_name)
