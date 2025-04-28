from scvi.train import TrainingPlan
from src.module._module import DCVAE


class CTLTrainingPlan(TrainingPlan):
    """constrastive training plan."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False
        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        """Training step."""
        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts
        # batch contains both data loader outputs
        total_loss_list = []
        kl_list = []
        rec_loss_list = []
        contra_loss_list = []

        # first input scRNA data to network
        for (key, tensor_list) in batch.items():
            loss_output_objs = []
            n_obs = 0
            zs = []
            qz = []
            qmu = []
            corr_ind = []
            for i, tensor in enumerate(tensor_list):
                n_obs += tensor[i].n_obs
                self.loss_kwargs.update({"kl_weight": self.kl_weight, "mode": i})
                inference_kwargs = {"mode": i}
                generative_kwargs = {"mode": i}
                inference_outputs, _, loss_output = self.forward(
                    tensor, # here the batch is tuple with 2 dict containing scRNA and ST data respectively
                    loss_kwargs=self.loss_kwargs,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                )
                zs.append(inference_outputs["z"])
                qz.append(inference_outputs["qz"])
                qmu.append(inference_outputs["qmu"])
                corr_ind.append(inference_outputs["corr_ind"])
                loss_output_objs.append(loss_output)

            contra_loss = DCVAE.contrast_loss(self.module, zs[0], zs[1], key, corr_ind)

            loss = sum([scl.loss for scl in loss_output_objs])

            CL_weight = self.module.CL_weight

            loss /= n_obs
            loss = loss + CL_weight*contra_loss
            rec_loss = sum([scl.reconstruction_loss_sum for scl in loss_output_objs])
            kl = sum([scl.kl_local_sum for scl in loss_output_objs])

            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()

            total_loss_list.append(loss)
            kl_list.append(kl)
            rec_loss_list.append(rec_loss)
            contra_loss_list.append(contra_loss)
            
        return_dict = {
            "loss": sum(total_loss_list),
            "reconstruction_loss_sum": sum(rec_loss_list),
            "kl_local_sum": sum(kl_list),
            "contrastive_loss": sum(contra_loss_list),
            "kl_global": 0.0
        }

        return return_dict

