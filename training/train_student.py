def train_student_with_kd(cfg, teacher):
    # similar pattern: load combined index with DAIC+E-DAIC+D-VLOG
    index = os.path.join(cfg['paths']['cache_root'], 'combined_index_student.csv')
    ds = WindowDataset(index)
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    device = torch.device(cfg['training']['device'])
    student = StudentModel().to(device)
    domain_disc = torch.nn.Sequential(torch.nn.Linear(512,128), torch.nn.ReLU(), torch.nn.Linear(128,3)).to(device)
    opt = torch.optim.Adam(list(student.parameters()) + list(domain_disc.parameters()), lr=cfg['training']['lr'])
    teacher.eval()
    for ep in range(cfg['training']['max_epochs_student']):
        student.train()
        losses=[]
        for b in dl:
            a = b['audio'].to(device).float(); v = b['visual'].to(device).float()
            y = torch.zeros(a.size(0), device=device)  # placeholder
            s_logit, s_phq, s_fused = student(a,v)
            with torch.no_grad():
                t_logit, t_phq, t_fused = teacher(a,v,None)
            loss_kd = kd_loss(s_logit, t_logit, T=cfg['losses']['kd']['T'], alpha=cfg['losses']['kd']['alpha'], y_true=y)
            # domain adaptation (example): GRL + MMD
            # domain_labels should come from b['meta']
            loss = loss_kd
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Student epoch {ep} loss {np.mean(losses):.4f}")
    torch.save(student.state_dict(), os.path.join(cfg['paths']['out_root'], 'student_kd.pt'))
    return student