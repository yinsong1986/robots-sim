# Security

Found a vulnerability? **Do not** open a public GitHub issue.

Amazon Web Services (AWS) coordinates responsible disclosure of
security vulnerabilities for this project.

Submit the issue privately to the AWS Vulnerability Disclosure Program:

- [HackerOne — AWS VDP](https://hackerone.com/aws_vdp)
- Email: <aws-security@amazon.com>

More details: [AWS Vulnerability Reporting](http://aws.amazon.com/security/vulnerability-reporting/).

The repo's [SECURITY.md](https://github.com/strands-labs/robots-sim/blob/main/SECURITY.md)
file is the canonical source.

## Threat model notes

`strands-robots-sim` is a thin plugin layer over Isaac Sim. The bulk of
the security surface — natural-language tool dispatch, mesh networking,
HuggingFace `trust_remote_code` gating, IoT bridging — lives upstream in
[`strands-labs/robots`](https://github.com/strands-labs/robots/blob/main/SECURITY.md).
File issues there for the agent / mesh / hardware paths.

What this repo adds to the threat model:

- **Isaac Sim Kit extensions.** Loading USD / URDF assets from untrusted
  sources runs through Omniverse Kit's importer pipeline. Treat untrusted
  USD files like untrusted code.
- **Replicator output paths.** `replicate(output_dir=...)` writes to
  arbitrary host paths. Validate the path against your dataset root in
  caller code; the engine does not sandbox it.
- **Nucleus URLs.** `IsaacConfig(nucleus_url=...)` and
  `STRANDS_ISAAC_NUCLEUS_URL` point Kit at a Nucleus server for asset
  resolution. A malicious Nucleus can serve crafted USDs that abuse the
  importer. Pin to trusted servers.

## Reporting checklist

When filing a private report, please include:

- Repo + tag / commit SHA you tested against.
- Repro snippet (Python script + which Isaac Sim version).
- Impact statement: what does the bug let an attacker do?
- Suggested fix (if any).
