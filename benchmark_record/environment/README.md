# Environment policy

This protocol archive does not contain a capture of the author's workstation or a
transient package solve. Such captures are local execution evidence, not stable
reader-facing protocol metadata, and `tools/package_record.py` excludes them by an
explicit allowlist.

Before definitive execution, each omicau and external-method environment must be
resolved from the version constraints in the protocol, locked, checksummed, and
associated with its run records. The result deposit must report the operating system,
Python or R version, package versions, hardware relevant to timing or determinism,
and the exact environment-lock digest. A change after freeze is handled under the
deviation policy.

Environment captures must not contain access tokens, credentials, local absolute
paths, raw data locations, subject identifiers, or other machine-private metadata.
