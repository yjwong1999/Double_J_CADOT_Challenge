# download resnext101-yolo12 trained without diffusion augmentation (synthetic data) and balanced sampling
curl -L -o "../models/resnext101-yolo12_naive_100epoch.pt" "https://www.dropbox.com/scl/fi/2lknmy8nbzkrxneooc6nh/resnext101-yolo12_naive_100epoch.pt?rlkey=cntg24np5tbbueq9qocy0kpgz&st=jiwc02ea&dl=0"

# download yolo12n trained with balanced sampling
curl -L -o "../models/yolo12n_balanced_100epoch.pt" "https://www.dropbox.com/scl/fi/o93jct4ufh5iwt9eidpw6/yolo12n_balanced_100epoch.pt?rlkey=mx6mj4ey9ioirzcm3zmpgn7zq&st=wrbr5e91&dl=0"

# download yolo12s trained with balanced sampling
curl -L -o "../models/yolo12s_balanced_50epoch.pt" "https://www.dropbox.com/scl/fi/zmvajy6y7tamkh5ewvk9y/yolo12s_balanced_50epoch.pt?rlkey=cnnzx0tgnb9z3ixi0q18ygsip&st=aul0cdqt&dl=0"

# download yolo12x trained with balanced sampling and diffusion augmentation (synthetic data)
curl -L -o "../models/yolo12x_balanced_aug_30epoch.pt" "https://www.dropbox.com/scl/fi/0sv2x92f03jbg1vs963d6/yolo12x_balanced_aug_30epoch.pt?rlkey=0q36jv6iqo722scebp38mwd0n&st=4k803qda&dl=0"

# download yolo12x trained with diffusion augmentation (synthetic data) only
curl -L -o "../models/yolo12x_aug_100epoch.pt" "https://www.dropbox.com/scl/fi/g1tuqpj5afc5vh3r3sefo/yolo12x_aug_100epoch.pt?rlkey=bqt0pot1g8nk5ubvz8buuw7c0&st=hzn69pia&dl=0"
