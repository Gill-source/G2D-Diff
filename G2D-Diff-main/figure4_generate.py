from rdkit import Chem
from rdkit.Chem import Draw, QED
from sascorer import calculateScore as calc_sa
import matplotlib.pyplot as plt

def generate_figure4(generator, genotype, class_id, num=4, device="cuda"):

    smiles_list = generator.generate_smiles(genotype, class_id, num=num)

    images = []
    texts = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        qed = QED.qed(mol)
        sa = calc_sa(mol)

        # predicted AUC
        auc = torch.sigmoid(
            generator.diffusion({
                "drug": torch.randn((1,128)).to(device),
                "class": torch.tensor([class_id]).to(device),
                "genotype": {k: v[:, :718] for k,v in genotype.items()}
            })
        ).item()

        images.append(Draw.MolToImage(mol, size=(300,300)))
        texts.append(f"SMILES={smi}\nQED={qed:.3f}, SA={sa:.2f}, AUC={auc:.3f}")

    # ------ Plot ------
    fig, axs = plt.subplots(1, len(images), figsize=(4*len(images), 4))

    for i, ax in enumerate(axs):
        ax.imshow(images[i])
        ax.axis("off")
        ax.set_title(texts[i], fontsize=8)

    plt.suptitle(f"G2D-Diff Generated Molecules (Class {class_id})", fontsize=16)
    plt.savefig(f"figure4_class_{class_id}.png", dpi=300)
    plt.show()
