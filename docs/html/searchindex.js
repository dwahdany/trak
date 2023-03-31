Search.setIndex({"docnames": ["bert", "clip", "index", "install", "modeloutput", "quickstart", "slurm", "trak"], "filenames": ["bert.rst", "clip.rst", "index.rst", "install.rst", "modeloutput.rst", "quickstart.rst", "slurm.rst", "trak.rst"], "titles": ["Applying <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAK</span></code> to a custom task #2: Text Classification using BERT", "Applying <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAK</span></code> to a custom task #3: CLIP", "TRAK: Attributing Model Behavior at Scale", "Installation FAQs", "Applying <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAK</span></code> to a custom task #1: Classification", "Quickstart \u2014 get <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAK</span></code> scores for <code class=\"code docutils literal notranslate\"><span class=\"pre\">CIFAR</span></code>", "Parallelize <code class=\"code docutils literal notranslate\"><span class=\"pre\">TRAK</span></code> scoring with <code class=\"code docutils literal notranslate\"><span class=\"pre\">SLURM</span></code>", "API reference"], "terms": {"In": [0, 1, 4, 5, 6, 7], "thi": [0, 1, 2, 3, 4, 5, 6, 7], "tutori": [0, 1, 4, 5, 7], "ll": [0, 1, 3, 4, 5, 6, 7], "go": [4, 6], "through": [], "process": [0, 5, 6], "appli": [2, 5, 7], "custom": [2, 3, 5, 7], "us": [1, 2, 3, 4, 5, 6, 7], "text": [1, 2, 4, 7], "classif": [1, 2, 5, 7], "an": [0, 1, 2, 3, 4, 5, 7], "exampl": [0, 1, 3, 6, 7], "check": [0, 1, 2, 3, 4, 5, 6, 7], "out": [0, 1, 4, 5, 6, 7], "textclassificationmodeloutput": [0, 7], "final": [1, 4, 5, 7], "result": [0, 1, 6, 7], "comput": [0, 1, 4, 6, 7], "attribut": [4, 5, 7], "score": [0, 1, 2, 4, 7], "requir": [3, 4, 6, 7], "defin": [1, 4, 5], "appropri": [0, 1, 4, 5], "guid": [2, 3], "intuit": [1, 4], "can": [0, 1, 2, 3, 4, 5, 6, 7], "just": [0, 1, 2, 4, 5], "think": [4, 6, 7], "deriv": [4, 7], "discuss": [], "multipl": [5, 6, 7], "binari": [0, 4], "multiclass": [4, 7], "variou": [], "nlp": 4, "etc": [0, 5, 7], "detail": [1, 2, 4, 5, 6, 7], "our": [0, 1, 2, 3, 4, 5, 6, 7], "paper": [0, 1, 2, 4, 6, 7], "short": [], "given": [0, 1, 4, 5, 7], "follow": [0, 1, 2, 3, 4, 6, 7], "train": [0, 1, 4, 6, 7], "set": [1, 4, 7], "": [0, 1, 2, 3, 4, 5, 6, 7], "z_1": [], "z_2": [], "ldot": [], "z_n": [], "where": [0, 4, 7], "each": [4, 5, 7], "z_i": [4, 7], "x_i": 1, "y_i": 1, "i": [0, 1, 2, 3, 4, 5, 6, 7], "input": [0, 1, 4, 7], "label": [0, 1, 4, 7], "pair": [1, 4, 7], "interest": [4, 7], "z": [0, 4, 7], "paramet": [4, 5, 7], "theta": [0, 4, 7], "implicitli": [], "repres": [], "machin": 3, "learn": [2, 4, 5, 6, 7], "some": [3, 4, 5, 6], "f": [0, 4, 5, 6, 7], "which": [0, 2, 3, 4, 5, 7], "map": [0, 6, 7], "real": 4, "number": [4, 7], "One": [], "e": [2, 3, 4, 5, 6, 7], "g": [2, 3, 4, 5, 6, 7], "cross": 4, "entropi": 4, "incur": 4, "addition": 7, "estim": [], "see": [0, 2, 3, 4, 5, 6, 7], "algorithm": [2, 5], "1": [0, 1, 2, 3, 5, 7], "n": [1, 5], "time": [1, 3], "diagon": 7, "matrix": 7, "one": [0, 1, 3, 4, 7], "minu": [], "correct": [0, 1, 4, 7], "class": [0, 4, 7], "probabl": [0, 1, 4, 7], "form": [0, 4], "come": [2, 3], "from": [0, 4, 5, 6, 7], "fact": 5, "log": [0, 1, 3, 4, 7], "left": [0, 1, 4, 6, 7], "frac": [0, 1, 4, 7], "p": [0, 1, 4, 5, 7], "right": [0, 1, 4, 7], "have": [1, 3, 4, 5, 6, 7], "ell": [0, 7], "wrt": [], "partial": [0, 4, 7], "exp": [0, 1, 4, 7], "If": [0, 1, 3, 4, 5, 7], "ar": [1, 2, 3, 4, 5, 7], "adapt": [0, 1, 2, 5], "more": [0, 2, 4, 5, 6, 7], "exot": [], "unrel": [], "might": 3, "modifi": 7, "accordingli": [], "all": [0, 1, 4, 5, 6, 7], "experi": 5, "wa": [0, 6], "necessari": 0, "provid": [0, 4, 5, 7], "dedic": 4, "abstractmodeloutput": [4, 5, 7], "take": [0, 2, 4, 5, 7], "care": 5, "translat": [], "torch": [4, 5, 6, 7], "modul": [0, 2, 3, 4, 5], "instanc": [4, 5, 6, 7], "achiev": 6, "via": [], "two": [1, 4], "method": [0, 1, 2, 4, 5, 6, 7], "get_output": [0, 1, 4, 7], "get_out_to_loss_grad": [0, 1, 4, 7], "particular": [1, 5, 7], "batch": [0, 1, 4, 5, 6, 7], "b": 7, "return": [0, 1, 4, 5, 7], "vector": [4, 7], "contain": [2, 3, 4, 7], "pass": [0, 1, 4, 5, 6, 7], "pytorch": [2, 7], "autograd": [4, 7], "sinc": [4, 7], "applic": 7, "so": [0, 4, 5, 7], "far": [], "could": 4, "analyt": [4, 7], "hardcod": 4, "thu": [4, 7], "avoid": [4, 7], "addit": [1, 4, 5, 7], "find": [0, 4, 6], "yourself": [0, 1, 4], "believ": [], "unlik": 1, "situat": 4, "t": [0, 1, 4, 6, 7], "structur": [5, 7], "well": 7, "when": [3, 4, 5, 6, 7], "onli": [1, 2, 5, 6, 7], "creat": [5, 7], "let": [1, 3, 5, 7], "assum": [1, 5, 7], "re": [1, 7], "familiar": [0, 1], "work": [1, 4], "rough": 1, "idea": [1, 6], "suffici": 1, "For": [0, 1, 3, 4, 5, 6, 7], "imag": [1, 4, 5, 7], "caption": [1, 7], "x": [0, 1, 2, 4, 5, 7], "y": [1, 4, 7], "denot": [], "embed": [1, 7], "phi": [1, 7], "psi": [1, 7], "pairwis": 1, "cosin": 1, "similar": [1, 5, 7], "between": [1, 5], "s_": 1, "ij": 1, "cdot": [1, 4, 7], "aim": 1, "maxim": 1, "ii": 1, "minim": [1, 4, 6, 7], "neq": [1, 7], "j": 1, "ell_": [], "sum_": [1, 7], "leq": 1, "ji": 1, "end": 0, "up": [], "choos": [1, 5], "section": [1, 3, 4, 7], "5": [1, 3, 5, 7], "why": [0, 4, 6], "good": 7, "f_": 1, "becaus": [], "choic": [0, 1, 4, 5, 7], "access": [1, 4], "util": [1, 5], "get_embed": [1, 7], "bit": [], "too": 7, "specif": [], "pai": [], "much": 7, "attent": [], "arrai": [1, 5, 6, 7], "all_img_embed": 1, "all_txt_embed": 1, "now": [1, 5, 6], "readi": [1, 5], "clipmodeloutput": [1, 7], "def": [0, 1, 4, 5, 6], "func_model": [0, 1, 4, 7], "weight": [0, 1, 4, 5, 7], "iter": [0, 1, 4, 5, 7], "tensor": [0, 1, 4, 7], "buffer": [0, 1, 4, 7], "image_embed": [1, 7], "text_embed": [1, 7], "_": [0, 1, 4, 7], "unsqueez": [0, 1, 4], "0": [0, 1, 2, 3, 4, 5, 6, 7], "ch": [0, 1, 4], "multinomi": 1, "arang": [0, 1, 4, 5], "float": [1, 5, 7], "num_sampl": [0, 1, 5, 7], "sim_b": 1, "replac": [1, 5], "fals": [0, 1, 4, 5, 7], "logsumexp": [0, 1, 4], "dim": [0, 1, 4], "sum": [0, 1, 4], "shape": [0, 1, 4, 5, 7], "should": [0, 1, 5, 6, 7], "remov": [1, 4], "extra": 1, "dimens": [1, 5, 7], "code": [0, 2, 5, 6, 7], "functorch": [4, 7], "vmap": [4, 7], "make": [2, 3, 4, 5, 7], "per": [4, 6, 7], "sampl": [4, 5, 7], "faster": [4, 5, 7], "parallel": [2, 5, 7], "about": [4, 6, 7], "rest": [], "simpli": [0, 4, 7], "equat": [], "earlier": 1, "simpl": [0, 6, 7], "self": [0, 1, 4, 5, 7], "temp": 1, "temperatur": [1, 7], "none": [0, 1, 7], "softmax": [0, 1, 4, 7], "diag": 1, "2": [1, 2, 4, 5, 7], "clone": [0, 1, 4], "detach": [0, 1, 4], "note": [0, 1, 3, 4, 5, 6, 7], "again": [0, 1, 5], "directli": [1, 4], "here": [0, 1, 3, 4, 5, 6, 7], "That": [0, 1, 4, 5, 6], "your": [1, 2, 3, 5, 7], "under": 2, "activ": 2, "develop": [2, 7], "we": [0, 1, 2, 4, 5, 6, 7], "still": [0, 2, 5], "version": [2, 3, 7], "lot": 2, "thing": [2, 4, 6], "mai": [2, 3, 7], "chang": [0, 2, 5], "base": [0, 2, 4, 7], "api": [0, 2, 5], "effect": [2, 4], "effici": 2, "data": [2, 7], "gradient": [0, 1, 2, 4, 7], "design": [2, 6, 7], "around": 2, "principl": 2, "eas": [2, 6], "you": [0, 1, 2, 3, 4, 5, 6, 7], "few": [2, 5], "line": [2, 5, 6], "quickstart": 2, "speed": 2, "fast": [2, 3], "cuda": [0, 1, 2, 5, 7], "kernel": [2, 3], "get": [1, 2, 3, 6, 7], "state": [2, 7], "art": 2, "bert": [2, 4], "qnli": [0, 2], "hour": 2, "8xa100": 2, "node": [2, 6], "simplic": 2, "lightweight": 2, "entir": [2, 5, 7], "codebas": 2, "less": 2, "than": [2, 7], "1000": 2, "It": [2, 7], "also": [0, 1, 2, 5, 7], "quit": 2, "modular": 2, "painless": 2, "ani": [2, 5, 7], "compon": 2, "need": [0, 1, 2, 3, 4, 5, 6, 7], "flexibl": 2, "task": [2, 5, 6, 7], "modal": 2, "easi": 2, "how": [0, 1, 2, 4, 5, 7], "clip": [0, 2, 4, 5, 7], "The": [0, 1, 2, 3, 4, 5, 6, 7], "packag": [2, 3], "pip": [2, 3], "traker": [0, 1, 2, 3, 4, 6], "To": [1, 2, 3, 5, 6], "faq": 2, "cifar": 2, "slurm": [2, 5, 7], "add": [3, 5], "subclass": 7, "modeloutput": 7, "refer": [2, 3, 4, 5], "index": [2, 3], "host": 3, "pypi": 3, "standard": [0, 3, 4], "jl": [3, 7], "project": [3, 5, 7], "step": [3, 5, 7], "compil": 3, "happen": [3, 7], "compat": [0, 3, 7], "below": [0, 3, 4, 5, 6], "tip": 3, "regard": 3, "10": [3, 5, 6], "pick": 3, "option": [3, 6, 7], "alreadi": [3, 4, 7], "been": 3, "setup": 3, "run": [0, 3, 7], "termin": [3, 6], "exist": [3, 7], "proce": 3, "unvers": 3, "compani": 3, "share": 3, "cluster": 3, "most": [3, 4, 7], "wai": [3, 5, 6], "enabl": [3, 7], "load": [0, 3, 6, 7], "tookit": 3, "without": [3, 7], "On": 3, "modulefil": 3, "command": [3, 6], "avail": [3, 7], "show": [0, 1, 3, 4, 5], "what": [3, 4, 6, 7], "doubt": 3, "plese": 3, "maintain": 3, "document": 3, "conda": 3, "c": [3, 7], "forg": 3, "cudatoolkit": 3, "dev": 3, "outdat": 3, "root": [3, 5], "feel": [3, 7], "confid": 3, "configur": 3, "nvidia": [3, 7], "instruct": 3, "http": [3, 7], "doc": [3, 6, 7], "com": [3, 7], "quick": 3, "start": [3, 5, 7], "html": [3, 7], "11": 3, "7": [3, 7], "12": 3, "oper": 3, "system": [3, 6], "preinstal": 3, "case": [0, 1, 3, 4, 5], "ship": [3, 4], "incompat": 3, "thei": 3, "usual": 3, "other": [1, 3], "alongsid": 3, "default": [0, 3, 4, 5, 6, 7], "ubuntu": 3, "repositori": 3, "sudo": 3, "apt": 3, "ppa": 3, "toolchain": 3, "r": [0, 3, 4, 7], "test": [3, 7], "updat": [3, 5, 7], "list": [3, 5, 7], "download": [3, 5], "befor": [3, 5, 7], "rune": 3, "ha": [3, 5, 7], "done": 3, "same": [0, 3, 5, 6, 7], "export": 3, "cxx": 3, "cc": 3, "quickli": 3, "correctli": 3, "python": [3, 6], "m": 3, "pytest": 3, "sv": 3, "test_rademach": 3, "py": [3, 6], "depend": [1, 3], "zsh": 3, "match": [1, 3], "found": 3, "sure": 3, "escap": 3, "squar": 3, "bracket": 3, "conceptu": 4, "actual": [], "loss": [0, 1, 4, 5, 7], "term": [1, 4, 6, 7], "classificaiton": [], "approproi": [], "first": [0, 1, 4, 5], "over": [4, 5], "Then": 4, "insid": [0, 1, 4, 7], "extend": [], "specifi": [4, 5, 6, 7], "want": [0, 4, 5, 7], "kind": 4, "evalu": [1, 4, 5], "formal": 4, "soft": [0, 4], "max": [0, 4], "motiv": [1, 4], "sever": 4, "order": [4, 5], "itself": 4, "w": [0, 4], "abov": [0, 1, 4, 5, 6, 7], "modelout": [], "q": 7, "resnet": 5, "9": [5, 6], "while": [1, 5], "architectur": [0, 5], "dataset": [0, 5], "easili": 5, "overal": [5, 6], "try": 5, "jupyt": 5, "notebook": [5, 7], "roughli": 5, "ten": 5, "minut": 5, "total": 5, "singl": [4, 5], "a100": [5, 6, 7], "gpu": [5, 6, 7], "either": [5, 7], "script": [0, 5, 6], "three": [0, 5, 6, 7], "save": [5, 7], "state_dict": [0, 1, 5, 7], "own": 5, "import": [0, 5, 6], "o": [5, 7], "pathlib": 5, "path": [5, 6, 7], "wget": 5, "tqdm": [0, 1, 5], "numpi": [5, 7], "np": 5, "amp": 5, "gradscal": 5, "autocast": 5, "nn": [0, 5, 7], "crossentropyloss": 5, "conv2d": 5, "batchnorm2d": 5, "optim": [4, 5], "sgd": 5, "lr_schedul": 5, "torchvis": 5, "resnet9": 5, "mul": 5, "__init__": [0, 5, 7], "super": [0, 5], "forward": [0, 4, 5, 7], "flatten": [5, 7], "view": [1, 5, 7], "size": [0, 4, 5, 7], "residu": 5, "construct_rn9": 5, "num_class": 5, "conv_bn": 5, "channels_in": 5, "channels_out": 5, "kernel_s": 5, "3": [0, 2, 4, 5, 7], "stride": 5, "pad": [0, 5], "group": 5, "sequenti": [5, 7], "bia": 5, "relu": 5, "inplac": 5, "true": [0, 5, 6, 7], "64": 5, "128": [0, 5], "256": 5, "maxpool2d": 5, "adaptivemaxpool2d": 5, "linear": 5, "get_dataload": 5, "batch_siz": [0, 5, 7], "num_work": 5, "8": [5, 6], "split": [0, 5], "transform": [0, 5, 7], "compos": 5, "randomhorizontalflip": 5, "randomaffin": 5, "totensor": 5, "normal": [5, 7], "4914": 5, "4822": 5, "4465": 5, "2023": 5, "1994": 5, "201": 5, "is_train": 5, "cifar10": 5, "tmp": 5, "loader": [5, 7], "dataload": [0, 5], "shuffl": [0, 5], "lr": 5, "4": [5, 7], "epoch": 5, "24": 5, "momentum": 5, "weight_decai": 5, "5e": 5, "lr_peak_epoch": 5, "label_smooth": 5, "opt": 5, "iters_per_epoch": 5, "len": [0, 5, 6], "cyclic": 5, "triangl": 5, "interp": 5, "schedul": [5, 6], "lambdalr": 5, "__getitem__": 5, "scaler": 5, "loss_fn": 5, "ep": 5, "rang": [0, 5, 7], "model_count": 5, "im": 5, "lab": 5, "enumer": [0, 5, 7], "zero_grad": 5, "set_to_non": 5, "scale": 5, "backward": [5, 7], "makedir": 5, "exist_ok": 5, "desc": [0, 1, 5], "memory_format": 5, "channels_last": 5, "loader_train": [0, 1, 5, 6], "512": 5, "sd_": 5, "pt": [5, 6], "remain": 5, "ckpt_file": 5, "rglob": 5, "ckpt": [5, 6], "map_loc": 5, "cpu": [5, 6], "ffcv": 5, "entri": [5, 7], "point": 5, "construct": 5, "call": [5, 6, 7], "argument": [0, 5], "doe": [5, 7], "initi": [5, 7], "do": [4, 5, 6, 7], "separ": 5, "string": [5, 7], "type": [5, 6, 7], "attribu": 5, "languag": [0, 5, 7], "style": 5, "contrast": [5, 7], "train_set_s": [0, 1, 5, 6, 7], "integ": 5, "keep": 5, "constructor": 5, "eval": [0, 5], "image_classif": [4, 5, 6, 7], "By": 5, "metadata": [5, 7], "store": [5, 7], "trak_result": [5, 7], "overrid": [5, 7], "save_dir": [0, 1, 5, 6, 7], "proj_dim": [0, 1, 5, 7], "2048": [5, 7], "curiou": 5, "correspond": [0, 1, 4, 5], "output": [5, 7], "random": [5, 7], "recommend": [5, 7], "000": 5, "40": 5, "object": [4, 5, 7], "determinist": 5, "model_id": [0, 1, 5, 6, 7], "associ": [0, 4, 5, 7], "uniqu": [5, 7], "load_checkpoint": [0, 1, 5, 6, 7], "tell": 5, "ve": 5, "inform": 5, "post": [5, 6], "next": [5, 7], "finalize_featur": [0, 1, 5, 6, 7], "mani": [5, 7], "altern": 5, "ind": [0, 5, 7], "instead": [0, 1, 4, 5, 6, 7], "length": 5, "indic": [5, 7], "within": 5, "significantli": [5, 7], "improv": [5, 6], "perform": [5, 6, 7], "But": [5, 6], "across": [5, 6, 7], "job": [5, 6], "differ": [0, 5, 6, 7], "purpos": [5, 7], "valid": [0, 5, 7], "loader_target": 5, "val": [0, 5], "As": [0, 1, 5], "start_scoring_checkpoint": [0, 1, 5, 6, 7], "num_target": [0, 1, 5, 7], "finalize_scor": [0, 1, 5, 6, 7], "function": [5, 6, 7], "prepar": [5, 7], "intern": [5, 7], "analog": [4, 5], "Be": 5, "did": 5, "wrong": 5, "silent": 5, "fail": 5, "know": 5, "clean": 5, "robust": 5, "hash": 5, "detect": 5, "open": 5, "issu": [5, 7], "github": 5, "assert": 5, "consist": 5, "onc": 5, "often": 6, "would": 6, "like": [4, 6, 7], "model": [6, 7], "help": 6, "mean": 6, "fortun": 6, "highli": 6, "paralleliz": 6, "sketch": [6, 7], "popular": 6, "skip": 6, "highlight": 6, "main": [0, 6, 7], "behind": 6, "write": [6, 7], "file": [6, 7], "featurize_and_scor": 6, "sbatch": 6, "mapreduc": 6, "reduc": [1, 6], "everyth": 6, "prior": 6, "argpars": 6, "argumentpars": 6, "loader_v": [0, 1, 6], "respect": [6, 7], "ckpt_": 6, "ds_train": [0, 6], "__name__": 6, "__main__": 6, "parser": 6, "add_argu": 6, "int": [6, 7], "arg": [0, 1, 6, 7], "parse_arg": 6, "bin": 6, "bash": 6, "gre": 6, "name": [0, 6, 7], "slurm_array_task_id": 6, "submit": 6, "u": [4, 6], "id": [0, 6, 7], "16": [0, 6], "don": [6, 7], "attempt": [6, 7], "current": [6, 7], "popul": [6, 7], "intermedi": [6, 7], "combin": 6, "prioriti": 6, "definit": 6, "problem": [1, 6, 7], "integr": 6, "distribut": 6, "str": 7, "load_from_save_dir": 7, "bool": 7, "devic": [0, 1, 4, 7], "abstractgradientcomput": 7, "functionalgradientcomput": 7, "abstractprojector": 7, "sourc": 7, "front": 7, "face": [0, 7], "readm": 7, "usag": 7, "union": 7, "ran": 7, "accept": 7, "text_classif": [0, 7], "implement": 7, "featur": [0, 1, 7], "directori": 7, "lead": 7, "flag": 7, "cudaprojector": 7, "rademach": 7, "give": [4, 7], "leav": 7, "notat": 7, "top": 7, "nabla_": 7, "must": 7, "del_grad": 7, "checkpoint": 7, "phi_c": 7, "A": 7, "exp_nam": 7, "target": [4, 7], "delet": 7, "npy": 7, "produc": 7, "uuid": 7, "gener": [4, 7], "init_projector": 7, "_allow_featurizing_already_regist": 7, "dictionari": [0, 7], "ti": 7, "behaviour": 7, "forbidden": 7, "regist": 7, "them": 7, "dict": 7, "modelout_fn": 7, "grad_dim": 7, "abc": 7, "gradientcomput": 7, "allow": 7, "behavior": 7, "load_model_param": 7, "load_state_dict": [0, 7], "compute_per_sample_grad": 7, "chosen": 7, "compute_loss_grad": 7, "scalar": 7, "everi": 7, "is_funct": 7, "determin": 7, "whether": 7, "abstract": 7, "noth": 7, "consid": [4, 7], "alon": 7, "grad": 7, "feasibl": 7, "someth": 7, "grad_out_to_loss": 7, "model_out_to_loss_grad": 7, "org": 7, "stabl": 7, "doesn": 7, "ad": 7, "signatur": [0, 7], "init": 7, "along": 7, "its": [4, 7], "make_functional_with_buff": 7, "iterativegradientcomput": 7, "out_to_loss": 7, "model_out_to_loss": 7, "leverag": 7, "henc": [0, 7], "slower": 7, "equival": 7, "veri": 7, "old": 7, "togeth": 7, "vision": 7, "imageclassificationmodeloutput": [4, 7], "iterimageclassificationmodeloutput": 7, "These": 7, "desir": 7, "logit": [0, 4, 7], "context": 7, "predict": 7, "simulated_batch_s": 7, "300": 7, "margin": [0, 4, 7], "multimod": 7, "rais": 7, "assertionerror": 7, "invok": 7, "logit_scal": 7, "simul": 7, "static": 7, "50000": 7, "embedding_dim": 7, "1024": [0, 1, 7], "preprocess_fn_img": 7, "preprocess_fn_txt": 7, "maximum": 7, "50_000": 7, "func": 7, "reweight": 7, "last": 7, "pleas": 7, "per_sample_grad": 7, "slightli": [0, 7], "break": 7, "shouldn": 7, "big": 7, "read": 7, "strongli": 7, "free": 7, "pr": 7, "fix": 7, "num_computed_embed": 7, "sim_batch_s": 7, "both": 7, "new": [0, 1, 4, 7], "part": 7, "confus": 7, "input_id": [0, 7], "token_type_id": [0, 7], "attention_mask": [0, 7], "seed": 7, "proj_typ": 7, "projectiontyp": 7, "hyperparamet": 7, "after": 7, "guearante": 7, "distanc": 7, "approxim": 7, "preserv": 7, "varieti": 7, "arxiv": 7, "ab": 7, "1411": 7, "2404": 7, "matric": 7, "iid": 7, "gaussian": 7, "includ": 7, "basicprojector": 7, "block_siz": 7, "200": 7, "dtype": 7, "float32": 7, "block": 7, "wise": 7, "accumul": 7, "larger": 7, "memori": 7, "footprint": 7, "capabl": 7, "generate_sketch_matrix": 7, "generator_st": 7, "get_generator_st": 7, "basicsingleblockprojector": 7, "float16": 7, "bare": 7, "bone": 7, "ineffici": 7, "matmul": 7, "even": 7, "resnet18": 7, "small": 7, "anyth": 7, "100": 7, "oom": 7, "unless": 7, "reason": 7, "cannot": 7, "kwarg": 7, "valu": 7, "enum": 7, "abstractsav": 7, "getter": 7, "setter": 7, "handl": 7, "record": 7, "dimension": 7, "reduct": 7, "johnson": 7, "lindenstrauss": 7, "current_": 7, "later": 7, "relat": 7, "clear_target_grad_count": 7, "reset": 7, "count": 7, "otherwis": 7, "load_stor": 7, "load_target_stor": 7, "mode": 7, "register_model_id": 7, "serialize_model_id_metadata": 7, "disk": 7, "commit": 7, "mmapsav": 7, "dure": 7, "init_stor": 7, "memmap": 7, "serial": 7, "dir": 7, "ye": 7, "error": 7, "mmap": 7, "modelidexcept": 7, "save_scor": 7, "except": [0, 7], "abstractscorecomput": 7, "scorecomput": 7, "get_xtx": 7, "get_x_xtx_inv": 7, "get_scor": 7, "target_grad": 7, "xtx": 7, "basicscorecomput": 7, "cuda_max_dim_s": 7, "100000": 7, "manner": 7, "100_000": 7, "basicsingleblockscorecomput": 7, "almost": 7, "get_num_param": 7, "get_params_dict": 7, "is_not_buff": 7, "params_dict": 7, "parameters_to_vector": 7, "reshap": 7, "peski": 7, "vectorize_and_ignore_buff": 7, "tupl": [0, 7], "grad_w0": 7, "grad_w1": 7, "grad_wp": 7, "grad_wi": 7, "num_param": 7, "librari": 4, "specfici": [], "l": [0, 4], "rare": 4, "explan": [0, 4], "bindex": [0, 4], "non_block": [0, 4], "logits_correct": [0, 4], "cloned_logit": [0, 4], "hacki": [], "inf": [0, 4], "reli": 4, "autodiff": 4, "loss_temperatur": [0, 4], "instanti": 4, "stateless": 4, "manual": [], "similalri": [], "automat": [1, 4], "differenti": [1, 4], "similarli": [0, 4], "element": [0, 4], "exclud": [0, 4], "ref": [], "anoth": [0, 1], "haven": [0, 1], "pre": 0, "langaug": 0, "huggingfac": 0, "finetun": 0, "question": 0, "answer": 0, "nli": 0, "automodelforsequenceclassif": 0, "fit": 0, "autoconfig": 0, "sequenceclassificationmodel": 0, "wrapper": 0, "sequenc": 0, "config": 0, "from_pretrain": 0, "num_label": 0, "finetuning_task": 0, "cache_dir": 0, "revis": 0, "use_auth_token": 0, "ignore_mismatched_s": 0, "sd": 0, "ckpt_path": 0, "et": [], "posit": 0, "keyword": 0, "hug": 0, "summari": [], "get_dataset": 0, "raw_dataset": 0, "load_dataset": 0, "glue": 0, "label_list": 0, "sentence1_kei": 0, "sentence2_kei": 0, "glue_task_to_kei": 0, "label_to_id": 0, "v": 0, "token": 0, "autotoken": 0, "use_fast": 0, "max_length": 0, "max_seq_length": 0, "preprocess_funct": 0, "els": 0, "truncat": 0, "load_from_cache_fil": 0, "train_dataset": 0, "d": 0, "eval_dataset": 0, "init_model": 0, "IF": 0, "TO": 0, "ON": 0, "full": 0, "5_000": 0, "val_set_s": [0, 1], "1_00": 0, "init_load": 0, "select": 0, "ds_val": 0, "collate_fn": 0, "default_data_col": 0, "reproduc": [0, 4], "redefin": 0, "classifci": 0, "math": [], "ident": 0, "lastli": 0, "saw": 0, "pretti": 0, "classifc": 0, "involv": 0, "notion": [0, 1], "process_batch": 0, "x_1": 1, "y_1": 1, "x_n": 1, "y_n": 1, "tri": 1, "align": 1, "vice": 1, "versa": 1, "observ": 1, "demonstr": 4, "customnmodeloutput": [], "custommodeloutput": 4, "natur": 4, "logist": 4, "regress": 4, "complet": 0, "l_": 1, "though": 4, "basic": 4, "illustr": 4}, "objects": {"": [[7, 0, 0, "-", "trak"]], "trak": [[7, 0, 0, "-", "gradient_computers"], [7, 0, 0, "-", "modelout_functions"], [7, 0, 0, "-", "projectors"], [7, 0, 0, "-", "savers"], [7, 0, 0, "-", "score_computers"], [7, 0, 0, "-", "traker"], [7, 0, 0, "-", "utils"]], "trak.gradient_computers": [[7, 1, 1, "", "AbstractGradientComputer"], [7, 1, 1, "", "FunctionalGradientComputer"], [7, 1, 1, "", "IterativeGradientComputer"]], "trak.gradient_computers.AbstractGradientComputer": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "compute_loss_grad"], [7, 2, 1, "", "compute_per_sample_grad"], [7, 3, 1, "", "is_functional"], [7, 2, 1, "", "load_model_params"]], "trak.gradient_computers.FunctionalGradientComputer": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "compute_loss_grad"], [7, 2, 1, "", "compute_per_sample_grad"], [7, 2, 1, "", "load_model_params"]], "trak.gradient_computers.IterativeGradientComputer": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "compute_loss_grad"], [7, 2, 1, "", "compute_per_sample_grad"], [7, 3, 1, "", "is_functional"], [7, 2, 1, "", "load_model_params"]], "trak.modelout_functions": [[7, 1, 1, "", "AbstractModelOutput"], [7, 1, 1, "", "CLIPModelOutput"], [7, 1, 1, "", "ImageClassificationModelOutput"], [7, 1, 1, "", "IterImageClassificationModelOutput"], [7, 1, 1, "", "TextClassificationModelOutput"]], "trak.modelout_functions.AbstractModelOutput": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "get_out_to_loss_grad"], [7, 2, 1, "", "get_output"]], "trak.modelout_functions.CLIPModelOutput": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "get_embeddings"], [7, 2, 1, "", "get_out_to_loss_grad"], [7, 2, 1, "", "get_output"], [7, 3, 1, "", "image_embeddings"], [7, 3, 1, "", "num_computed_embeddings"], [7, 3, 1, "", "sim_batch_size"], [7, 3, 1, "", "text_embeddings"]], "trak.modelout_functions.ImageClassificationModelOutput": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "forward"], [7, 2, 1, "", "get_out_to_loss_grad"], [7, 2, 1, "", "get_output"]], "trak.modelout_functions.IterImageClassificationModelOutput": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "get_out_to_loss_grad"], [7, 2, 1, "", "get_output"]], "trak.modelout_functions.TextClassificationModelOutput": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "forward"], [7, 2, 1, "", "get_out_to_loss_grad"], [7, 2, 1, "", "get_output"]], "trak.projectors": [[7, 1, 1, "", "AbstractProjector"], [7, 1, 1, "", "BasicProjector"], [7, 1, 1, "", "BasicSingleBlockProjector"], [7, 1, 1, "", "CudaProjector"], [7, 1, 1, "", "ProjectionType"]], "trak.projectors.AbstractProjector": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "project"]], "trak.projectors.BasicProjector": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "generate_sketch_matrix"], [7, 2, 1, "", "get_generator_states"], [7, 2, 1, "", "project"]], "trak.projectors.BasicSingleBlockProjector": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "generate_sketch_matrix"], [7, 2, 1, "", "project"]], "trak.projectors.CudaProjector": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "project"]], "trak.projectors.ProjectionType": [[7, 3, 1, "", "normal"], [7, 3, 1, "", "rademacher"]], "trak.savers": [[7, 1, 1, "", "AbstractSaver"], [7, 1, 1, "", "MmapSaver"], [7, 4, 1, "", "ModelIDException"]], "trak.savers.AbstractSaver": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "clear_target_grad_count"], [7, 2, 1, "", "del_grads"], [7, 2, 1, "", "load_store"], [7, 2, 1, "", "load_target_store"], [7, 2, 1, "", "register_model_id"], [7, 2, 1, "", "serialize_model_id_metadata"]], "trak.savers.MmapSaver": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "clear_target_grad_count"], [7, 2, 1, "", "del_grads"], [7, 2, 1, "", "init_store"], [7, 2, 1, "", "load_store"], [7, 2, 1, "", "load_target_store"], [7, 2, 1, "", "register_model_id"], [7, 2, 1, "", "save_scores"], [7, 2, 1, "", "serialize_model_id_metadata"]], "trak.score_computers": [[7, 1, 1, "", "AbstractScoreComputer"], [7, 1, 1, "", "BasicScoreComputer"], [7, 1, 1, "", "BasicSingleBlockScoreComputer"]], "trak.score_computers.AbstractScoreComputer": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "get_scores"], [7, 2, 1, "", "get_x_xtx_inv"], [7, 2, 1, "", "get_xtx"]], "trak.score_computers.BasicScoreComputer": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "get_scores"], [7, 2, 1, "", "get_x_xtx_inv"], [7, 2, 1, "", "get_xtx"]], "trak.score_computers.BasicSingleBlockScoreComputer": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "get_scores"], [7, 2, 1, "", "get_x_xtx_inv"], [7, 2, 1, "", "get_xtx"]], "trak.traker": [[7, 1, 1, "", "TRAKer"]], "trak.traker.TRAKer": [[7, 2, 1, "", "__init__"], [7, 2, 1, "", "featurize"], [7, 2, 1, "", "finalize_features"], [7, 2, 1, "", "finalize_scores"], [7, 2, 1, "", "init_projector"], [7, 2, 1, "", "load_checkpoint"], [7, 2, 1, "", "score"], [7, 2, 1, "", "start_scoring_checkpoint"]], "trak.utils": [[7, 5, 1, "", "get_num_params"], [7, 5, 1, "", "get_params_dict"], [7, 5, 1, "", "is_not_buffer"], [7, 5, 1, "", "parameters_to_vector"], [7, 5, 1, "", "vectorize_and_ignore_buffers"]]}, "objtypes": {"0": "py:module", "1": "py:class", "2": "py:method", "3": "py:attribute", "4": "py:exception", "5": "py:function"}, "objnames": {"0": ["py", "module", "Python module"], "1": ["py", "class", "Python class"], "2": ["py", "method", "Python method"], "3": ["py", "attribute", "Python attribute"], "4": ["py", "exception", "Python exception"], "5": ["py", "function", "Python function"]}, "titleterms": {"add": [], "task": [0, 1, 4], "traker": [5, 7], "subclass": [], "modeloutput": [], "bert": 0, "base": [], "clip": 1, "prelud": 4, "model": [0, 1, 2, 4, 5], "output": [0, 1, 4], "function": [0, 1, 4], "The": [], "modelout": [], "loss": [], "gradient": [], "q": 3, "term": [], "how": 3, "we": [], "implement": [0, 1, 4], "trak": [0, 1, 2, 3, 4, 5, 6, 7], "what": [], "you": [], "need": [], "do": [], "new": [], "attribut": 2, "behavior": 2, "scale": 2, "overview": [1, 2], "instal": [2, 3], "content": [2, 7], "indic": 2, "tabl": 2, "faq": 3, "nvcc": 3, "cuda": 3, "toolkit": 3, "gcc": 3, "verifi": 3, "work": 3, "misc": 3, "A": 3, "appli": [0, 1, 4], "custom": [0, 1, 4], "text": 0, "classif": [0, 4], "exampl": [4, 5], "quickstart": 5, "get": 5, "score": [5, 6], "cifar": 5, "load": 5, "checkpoint": [5, 6], "set": 5, "up": 5, "class": 5, "comput": 5, "featur": [5, 6], "train": 5, "data": [0, 5], "target": 5, "parallel": 6, "slurm": 6, "1": [4, 6], "each": 6, "2": [0, 6], "run": 6, "3": [1, 6], "gather": 6, "final": 6, "api": 7, "refer": 7, "modul": 7, "gradient_comput": 7, "modelout_funct": 7, "projector": 7, "saver": 7, "score_comput": 7, "util": 7, "bonu": [], "extend": [0, 4], "other": [0, 4], "abstractmodeloutput": [], "put": [0, 1], "togeth": [0, 1], "us": 0}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 8, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.todo": 2, "sphinx.ext.viewcode": 1, "sphinx": 57}, "alltitles": {"Installation FAQs": [[3, "installation-faqs"]], "How to install TRAK?": [[3, "how-to-install-trak"]], "How to install nvcc (CUDA toolkit)?": [[3, "how-to-install-nvcc-cuda-toolkit"]], "How to install gcc?": [[3, "how-to-install-gcc"]], "Verify that the installation worked": [[3, "verify-that-the-installation-worked"]], "Misc Q&A": [[3, "misc-q-a"]], "Parallelize TRAK scoring with SLURM": [[6, "parallelize-trak-scoring-with-slurm"]], "1. Featurizing each checkpoint": [[6, "featurizing-each-checkpoint"]], "2. Run featurize in parallel": [[6, "run-featurize-in-parallel"]], "3. Gather final scores": [[6, "gather-final-scores"]], "API reference": [[7, "api-reference"]], "trak.traker module": [[7, "module-trak.traker"]], "trak.gradient_computers module": [[7, "module-trak.gradient_computers"]], "trak.modelout_functions module": [[7, "module-trak.modelout_functions"]], "trak.projectors module": [[7, "module-trak.projectors"]], "trak.savers module": [[7, "module-trak.savers"]], "trak.score_computers module": [[7, "module-trak.score_computers"]], "trak.utils module": [[7, "module-trak.utils"]], "Module contents": [[7, "module-trak"]], "Quickstart \u2014 get TRAK scores for CIFAR": [[5, "quickstart-get-trak-scores-for-cifar"]], "Load model checkpoints": [[5, "load-model-checkpoints"]], "Set up the TRAKer class": [[5, "set-up-the-traker-class"]], "Compute TRAK features for training data": [[5, "compute-trak-features-for-training-data"]], "Compute TRAK scores for target examples": [[5, "compute-trak-scores-for-target-examples"]], "TRAK: Attributing Model Behavior at Scale": [[2, "trak-attributing-model-behavior-at-scale"]], "Overview": [[2, "overview"]], "Install": [[2, "install"]], "Contents": [[2, "contents"]], "Indices and tables": [[2, "indices-and-tables"]], "Putting it together": [[0, "putting-it-together"], [1, "putting-it-together"]], "Implementing the model output function": [[0, "implementing-the-model-output-function"], [1, "implementing-the-model-output-function"]], "Applying TRAK to a custom task #2: Text Classification using BERT": [[0, "applying-trak-to-a-custom-task-2-text-classification-using-bert"]], "Model and Data": [[0, "model-and-data"]], "Text Classification": [[0, "text-classification"]], "Extending to other tasks": [[0, "extending-to-other-tasks"], [4, "extending-to-other-tasks"]], "Applying TRAK to a custom task #3: CLIP": [[1, "applying-trak-to-a-custom-task-3-clip"]], "CLIP overview": [[1, "clip-overview"]], "Applying TRAK to a custom task #1: Classification": [[4, "applying-trak-to-a-custom-task-1-classification"]], "Prelude: Model output functions": [[4, "prelude-model-output-functions"]], "Implementing model output functions in TRAK": [[4, "implementing-model-output-functions-in-trak"]], "Example: Classification": [[4, "example-classification"]], "Implementation": [[4, "implementation"]]}, "indexentries": {}})