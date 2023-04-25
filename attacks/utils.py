
import torch.nn.functional as F
from tqdm import tqdm, trange
import torch


def set_robustness_attack_config(attack_norm, attack_epsilon, attack_alpha, attack_iters, attack_random_restarts,
                                 targeted):
    if attack_norm == 'l2':
        constraint = '2'
    elif attack_norm == 'linf':
        constraint = 'inf'
    elif attack_norm == 'none':
        constraint = 'unconstrained'
    elif attack_norm == 'fourier':
        constraint = 'fourier'
    else:
        constraint = 'unconstrained'
    attack_kwargs = {
        'constraint': constraint,  # use L2-PGD
        'eps': attack_epsilon,  # L2 radius around original image
        'step_size': attack_alpha,
        'iterations': attack_iters,
        'targeted': targeted,
        'random_restarts': attack_random_restarts,
        'do_tqdm': False,
    }
    return attack_kwargs


def conduct_pgd(attack_iters_lst, attack_epsilon_lst, attack_norm_lst, attack_alpha_lst, pgd_random_restarts,
                test_attack_loader, augmenter, attack_targeted):
    device = augmenter.device
    attack_losses = {}
    attack_accs = {}
    for attack_norm_i in trange(len(attack_norm_lst), desc='attack norm loop'):
        attack_norm = attack_norm_lst[attack_norm_i]
        attack_epsilon_lst_i = attack_epsilon_lst[attack_norm_i]
        attack_alpha_lst_i = attack_alpha_lst[attack_norm_i]

        attack_losses[attack_norm] = {}
        attack_accs[attack_norm] = {}
        for attacks_iter_i in trange(len(attack_iters_lst), desc='attack iter loop'):
            attack_iter = attack_iters_lst[attacks_iter_i]
            attack_losses[attack_norm][attack_iter] = {}
            attack_accs[attack_norm][attack_iter] = {}
            for attacks_eps_i in trange(len(attack_epsilon_lst_i),
                                        desc='attack epsilon loop with {} iter'.format(
                                            attack_iter)):
                attack_epsilon = attack_epsilon_lst_i[attacks_eps_i]
                attack_alpha = attack_alpha_lst_i[attacks_eps_i]
                robustness_attack_config = set_robustness_attack_config(attack_norm,
                                                                        attack_epsilon,
                                                                        attack_alpha,
                                                                        attack_iter,
                                                                        pgd_random_restarts,
                                                                        attack_targeted)
                print('\nStart attacking with epsilon : {}\talpha : {} \t iter : {}...'.format(
                    attack_epsilon,
                    attack_alpha,
                    attack_iter))

                count_total = 0.
                correct_total = 0.
                detached_loss_total = 0.

                # for sorted_ix, batch in enumerate(tqdm(test_attack_loader, desc='attack loop')):
                for test_i, batch in enumerate(
                        tqdm(test_attack_loader, desc='attack loop (val)')):
                    # setting model to eval mode
                    augmenter.set_eval()

                    X, y = batch
                    if attack_targeted:
                        # shift the labels up.
                        y_target = y + 1
                        y_target %= (test_attack_loader.dataset.classes.__len__() - 1)
                    else:
                        y_target = y
                    X, y, y_target = X.to(device), y.to(device), y_target.to(device)
                    X_adv = augmenter.adv_example(X, y_target, robustness_attack_config)

                    # calc the loss on adv. examples
                    with torch.no_grad():
                        outputs = augmenter.eval_mode_pred(X_adv)
                        val_loss = augmenter.calc_eval_loss_from_output(outputs, y)

                    _, predictions = torch.max(outputs.data, 1)
                    count_total += y.size(0)
                    detached_loss_total += val_loss
                    correct_total += predictions.eq(y.data).sum().float().cpu().numpy().item()

                attack_loss = detached_loss_total / count_total
                attack_acc = 100. * correct_total / count_total

                print('Finished attacking with epsilon : {} \t iter : {}...\n'.format(
                    attack_epsilon,
                    attack_iter))
                print('Attack Test Loss: {:.5f}, Acc: {:.5f}'.format(attack_loss, attack_acc))

                attack_losses[attack_norm][attack_iter][attack_epsilon] = attack_loss
                attack_accs[attack_norm][attack_iter][attack_epsilon] = attack_acc

    return attack_losses, attack_accs
