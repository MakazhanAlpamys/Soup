# Security Policy

## Supported Versions

We provide security updates for the following versions:

- **Latest minor version:** Active support (e.g., v0.15.x)
- **Previous minor versions:** Bug-fix support only
- **Versions older than 3 minor versions:** No support

Example:
- v0.15.0-0.15.x → Full support (latest)
- v0.14.0-0.14.x → Bug-fix support only
- v0.13.x and below → No support

## Reporting a Vulnerability

**Do not** open a public issue or pull request for security vulnerabilities.

Instead, use [GitHub Security Advisories](https://github.com/MakazhanAlpamys/Soup/security/advisories/new) to report privately, or email **vpn.alpamys@gmail.com** with:

1. **Description**: A clear explanation of the vulnerability
2. **Steps to Reproduce**: How to trigger or demonstrate the issue
3. **Affected Versions**: Which Soup versions are impacted
4. **Suggested Fix** (optional): Any proposed solutions
5. **Contact Info**: Your email for follow-up (optional)

### What to Include

```
To: vpn.alpamys@gmail.com
Subject: Security Vulnerability Report: [Brief Title]

Description:
[Explain the vulnerability in detail]

Affected Component:
[e.g., data/loader.py, trainer/sft.py, etc.]

Steps to Reproduce:
1. [Step 1]
2. [Step 2]
3. ...

Impact:
[What could go wrong? Data exposure? RCE? DoS?]

Suggested Fix (optional):
[Your proposed solution, if any]
```

## Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: 1-3 business days
- **Fix Development**: Varies by severity
- **Patch Release**: As soon as possible after fix verification
- **Public Disclosure**: Coordinated with reporter (typically 90 days after patch release)

## Severity Levels

- **Critical**: Remote code execution, data exposure, complete compromise (patch within 24-48 hours)
- **High**: Authentication bypass, privilege escalation, denial of service (patch within 1 week)
- **Medium**: Information disclosure, partial compromise (patch within 2 weeks)
- **Low**: Minor issues with limited impact (patch in next regular release)

## Security Best Practices

When using Soup, follow these practices to stay secure:

### 1. Keep Soup Updated

```bash
pip install --upgrade soup-cli
```

### 2. Protect API Keys

Never commit API keys or secrets to version control. Use environment variables:

```bash
export HUGGINGFACE_TOKEN=your_token_here
export WANDB_API_KEY=your_key_here
soup train
```

### 3. Validate Data

- Only use trusted datasets
- Verify checksums for large datasets
- Inspect data for malicious content before training

### 4. Model Permissions

- Be cautious when downloading models from untrusted sources
- Use model hub providers with verified publishers (HuggingFace, Meta, etc.)
- Keep track of which models you've fine-tuned and their base model sources

### 5. GPU/Compute Safety

- Run on isolated machines if training on sensitive data
- Clear cache and temporary files after training
- Don't share fine-tuned models containing sensitive information

## Known Vulnerabilities

We maintain a log of known security issues and their fixes. This will be updated as issues are discovered and resolved.

### Current Status

No known critical vulnerabilities in current releases.

### Security Hardening History

- **v0.10.10**: Bearer token auth on Web UI, CORS restrictions, path traversal protection, SSRF prevention, max_tokens limits, supply-chain pinning (llama.cpp b5270), deprecated CLI secret flags
- **v0.12.0**: experiment_name path traversal validation, GaLore parameter type enforcement
- **v0.13.0**: Batch inference max_tokens capped at 16384, trust_remote_code warning
- **v0.14.0**: Plaintext loader UTF-8 encoding, MoE config validation (moe_aux_loss_coeff ge=0, moe_lora boolean only)
- **v0.14.3**: Data validate auto-detects format, Web UI `--show-token` flag + auth token documented
- **v0.15.0**: `rope_scaling_type` Literal constraint, `max_length` bounds (ge=64, le=1048576), FSDP config key allowlist, Liger Kernel exception handling narrowed

## Security Scanning

- All code is scanned with `ruff` for style and common issues
- Dependencies are regularly updated to patch known CVEs
- GitHub's dependency scanning alerts us to vulnerable dependencies
- We use GitHub Actions CI/CD for continuous integration

## Dependency Updates

We actively monitor and update dependencies:

- Major dependency updates: Tested in PR before merging
- Security patches: Applied immediately and released as patch versions
- Deprecated dependencies: Replaced proactively

## Coming Soon

- [x] Automated dependency scanning
- [ ] SBOM (Software Bill of Materials) for each release
- [ ] Third-party security audit (after 1.0.0 release)

## Questions?

If you have security questions (not vulnerability reports) or need clarification:

- Open a GitHub Discussion tagged `security`
- Open a [GitHub Issue](https://github.com/MakazhanAlpamys/Soup/issues) (non-vulnerability inquiries)
- Check our [CONTRIBUTING.md](CONTRIBUTING.md) for general support

## License

This Security Policy is provided under the MIT license, same as the Soup project.

---

**Last Updated**: March 2026

For the latest version of this policy, visit: https://github.com/MakazhanAlpamys/Soup/blob/main/SECURITY.md
