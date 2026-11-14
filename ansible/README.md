# LightLLM hot-standby failover

Ansible roles, site playbook, and validator for deploying a LightLLM gateway in a
hot-standby failover arrangement (Issue #55).

## Roles

- `lightllm_primary` — deploys and activates the primary gateway (port 40000).
- `lightllm_standby` — deploys and activates the standby gateway (port 40001).

Both install LightLLM into a shared virtualenv, render a server config + a
systemd unit, and start/enable the service. The standby is promoted (traffic
switched to it) by an operator/controlller, not by Ansible itself.

## Playbook

    ansible-playbook -i inventory.example site.yml

Runs the primary play then the standby play. Failover bookkeeping (promoting the
standby and demoting the primary) is an operational step, not part of this role
tree — see the deployment README for the exact handoff steps.

## Validation

No ansible is required. The tree is validated with PyYAML:

    python3 ansible/validate.py

The validator checks:
- every YAML role file parses and has its required keys (`defaults`, `vars`,
  `tasks`, `handlers`, `meta`);
- `site.yml` is a non-empty list of plays, each with `name` and `hosts`;
- `inventory.example` carries the `[lightllm_primary]` and `[lightllm_standby]`
  group headers;
- every expected Jinja template exists (templates are rendered, not YAML, so they
  are existence-checked only).

## Layout

    ansible/
    ├── roles/
    │   ├── lightllm_primary/{defaults,vars,tasks,handlers,meta,templates}/
    │   └── lightllm_standby/{defaults,vars,tasks,handlers,meta,templates}/
    ├── site.yml
    ├── inventory.example
    ├── validate.py
    └── README.md
