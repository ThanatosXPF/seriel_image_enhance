import os
from imageio import imread
import numpy as np
import config as c
from util import construct_path, construct_save_ind, normalize_frames


class Iterator:
    def __init__(self, start_time, end_time, mode="train"):
        self.start_time = start_time
        self.end_time = end_time

        self.times = self.gen_times()
        self.visual_num = c.VISUAL_NUM
        self.batch = c.BATCH_SIZE
        self.time_seq = c.TIME_SEQ

        if c.DATA_PRECISION == "HALF":
            self.dtype = np.float16
        elif c.DATA_PRECISION == "SINGLE":
            self.dtype = np.float32

        if mode == "valid":
            self.save_ind = construct_save_ind(self.times, self.visual_num)

    def gen_times(self):
        time_list = []
        times = []
        for PATH in c.DATASET_DIR:
            times += sorted(os.listdir(PATH))

        for t in times:
            if self.start_time <= t <= self.end_time:
                time_list.append(construct_path(t))

        return time_list

    def read_image(self, paths, b_i, data):
        for i, p in enumerate(paths):
            img = imread(p)
            img = np.reshape(img, (c.H, c.W, 1))
            data[b_i, i,:,:,:] = img

    def random_sample(self):
        targets = np.random.choice(self.times, self.batch)
        pred_data = np.zeros([c.BATCH_SIZE, c.TIME_SEQ, c.H, c.W, 1], dtype=self.dtype)
        gt_data = np.zeros([c.BATCH_SIZE, c.TIME_SEQ, c.H, c.W, 1], dtype=self.dtype)
        for b, target in enumerate(targets):
            out_path = os.path.join(target, "out")
            pred_path = os.path.join(target, "pred")

            for i in range(1, c.TIME_SEQ+1):
                data = imread(os.path.join(pred_path, "{}.png".format(i)))
                data = np.reshape(data, (c.H, c.W, 1))
                pred_data[b, i-1, :,:,:] = data

                data = imread(os.path.join(out_path, "{}.png".format(i)))
                data = np.reshape(data, (c.H, c.W, 1))
                gt_data[b, i-1,:,:,:] = data
        if c.NORMALIZE:
            pred_data = normalize_frames(pred_data)
            gt_data = normalize_frames(gt_data)

        if not c.COLD_START:
            gru_in_data = np.zeros([c.BATCH_SIZE, c.IN_SEQ, c.H, c.W, 1], dtype=self.dtype)
            for b, target in enumerate(targets):
                g_i_path = os.path.join(target, "in")
                paths = []
                for i in range(c.TOTAL_IN - c.IN_SEQ, c.TOTAL_IN):
                    paths.append(os.path.join(g_i_path, "{}.png".format(i)))
                self.read_image(paths, b, gru_in_data)

            pred_data = np.concatenate([gru_in_data, pred_data], axis=1)
            gt_data = np.concatenate([gru_in_data, gt_data], axis=1)
        return pred_data, gt_data

    def sequent_sample(self):
        for t, v in zip(self.times, self.save_ind):
            pred_data = np.zeros([c.BATCH_SIZE, c.TIME_SEQ, c.H, c.W, 1], dtype=self.dtype)
            gt_data = np.zeros([c.BATCH_SIZE, c.TIME_SEQ, c.H, c.W, 1], dtype=self.dtype)
            out_path = os.path.join(t, "out")
            pred_path = os.path.join(t, "pred")

            for i in range(1, c.TIME_SEQ+1):
                data = imread(os.path.join(pred_path, "{}.png".format(i)))
                data = np.reshape(data, (c.H, c.W, 1))
                pred_data[:, i - 1, :, :, :] = data

                data = imread(os.path.join(out_path, "{}.png".format(i)))
                data = np.reshape(data, (c.H, c.W, 1))
                gt_data[:, i - 1, :, :, :] = data

            if c.NORMALIZE:
                pred_data = normalize_frames(pred_data)
                gt_data = normalize_frames(gt_data)

            if not c.COLD_START:
                gru_in_data = np.zeros([c.BATCH_SIZE, c.IN_SEQ, c.H, c.W, 1], dtype=self.dtype)
                g_i_path = os.path.join(t, "in")
                paths = []
                for i in range(c.TOTAL_IN - c.IN_SEQ, c.TOTAL_IN):
                    paths.append(os.path.join(g_i_path, "{}.png".format(i)))
                for i in range(c.BATCH_SIZE):
                    self.read_image(paths, i, gru_in_data)
                pred_data = np.concatenate([gru_in_data, pred_data], axis=1)
                gt_data = np.concatenate([gru_in_data, gt_data], axis=1)
            yield pred_data, gt_data, t, v