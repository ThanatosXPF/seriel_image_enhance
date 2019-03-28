import os
import logging

from model import Model
from iterator import Iterator
import config as c
from util import config_log, save_png


class CAEGruRunner:
    def __init__(self, path):
        self.model = Model(path)
        if not path:
            self.model.initialize()

    def train(self):
        iterator = Iterator("201401010000", "201808010000")
        iter = 1
        while iter < c.ITER:
            in_data, gt_data = iterator.random_sample()
            l2, mse, rmse, mae = self.model.train_step(in_data, gt_data)
            logging.info("Iter {}: \n\t l2: {} \n\t mse:{} \n\t rmse:{} \n\t mae:{}".format(iter, l2, mse, rmse, mae))
            if iter % c.SAVE_ITER == 0:
                self.model.save_model(iter)
            if iter % c.VALID_ITER == 0:
                self.valid(iter)
            iter += 1

    def valid(self, iter):
        iterator = Iterator("201808010000", "201812310000", mode="valid")
        num = 1
        t_l2 = 0
        t_mse = 0
        t_rmse = 0
        t_mae = 0
        count = 0
        for in_data, out_data, date, save in iterator.sequent_sample():
            l2, mse, rmse, mae, result = self.model.valid(in_data, out_data)
            t_mse += mse
            t_rmse += rmse
            t_mae += mae
            t_l2 += l2
            count += 1

            logging.info("Valid {} {}: \n\t l2:{} \n\t mse:{} \n\t ".format(num, date,l2,  mse) +
                         "rmse:{} \n\t mae:{}".format(rmse, mae))

            if save:
                logging.info("Save {} results".format(date))
                save_path = c.SAVE_VALID_DIR + str(iter) + "/" + date[-12:] + "/"

                path = os.path.join(save_path, "in")
                save_png(in_data[0], path)

                path = os.path.join(save_path, "out")
                save_png(result[0], path)

                path = os.path.join(save_path, "gt")
                save_png(out_data[0], path)
            num += 1

        t_mse /= count
        t_rmse /= count
        t_mae /= count
        t_l2 /= count
        logging.info("Valid in {}: \n\t l2 \n\t mse:{} \n\t ".format(iter,t_l2, t_mse) +
                     "rmse:{} \n\t mae:{}".format(t_rmse, t_mae))
        logging.info("#"*30)

    def test(self, iter):
        iterator = Iterator("201808010000", "201812310000", mode="valid")
        for in_data, out_data, date, save in iterator.sequent_sample():
            *_, result = self.model.valid(in_data, out_data)
            logging.info("Save {} results".format(date))
            save_path = c.SAVE_TEST_DIR + str(iter) + "/" + date[-12:] + "/"
            print(save_path)

            path = os.path.join(save_path, "in")
            save_png(in_data[0], path)

            path = os.path.join(save_path, "pred")
            save_png(result[0], path)

            path = os.path.join(save_path, "out")
            save_png(out_data[0], path)


if __name__ == '__main__':
    config_log()
    # path = "/extend/sml_data/output/0311_1418_origin/Save/model.ckpt.95000"
    path = None
    runner = CAEGruRunner(path)
    runner.train()
