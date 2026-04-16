from train_gpt import Hyperparameters, GPT
args = Hyperparameters()
base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        seq_len=args.train_seq_len
    )

print(sum([p.numel() for p in base_model.parameters()]))