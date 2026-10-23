"""Self-signed SSL certificate helpers.

HTTPS is required for browsers to grant microphone / camera / screen-share
access from any origin other than ``localhost``. To let other PCs on the LAN
use their own mic and camera, the server must be served over HTTPS. This module
generates a self-signed certificate covering this machine's hostname and all of
its local IP addresses so the certificate keeps working even when the IP changes.
"""

import socket
import ipaddress
import datetime
from pathlib import Path

from loguru import logger


def _local_ip_addresses() -> list[str]:
    """Best-effort collection of this machine's IPv4 addresses."""
    addresses: set[str] = {"127.0.0.1"}
    hostname = socket.gethostname()
    try:
        for info in socket.getaddrinfo(hostname, None):
            addr = info[4][0]
            # skip IPv6 link-local clutter; keep plain IPv4/IPv6
            addresses.add(addr)
    except socket.gaierror:
        pass

    # The getaddrinfo trick can miss the active LAN interface; probe via a
    # dummy UDP socket (no packets are actually sent).
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addresses.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass

    return sorted(addresses)


def _uncovered_addresses(cert_path: Path, extra: tuple[str, ...] = ()) -> list[str]:
    """Which of this machine's current IPs the existing certificate omits.

    An unreadable or malformed certificate counts as covering nothing, so it is
    replaced rather than served — a cert that cannot be parsed here is one the
    browser is unlikely to accept either.
    """
    try:
        from cryptography import x509

        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        covered = {str(v) for v in san.get_values_for_type(x509.IPAddress)}
        covered |= {str(v) for v in san.get_values_for_type(x509.DNSName)}
    except Exception as e:  # noqa: BLE001 — an unusable cert is a missing cert
        logger.warning(f"Could not read the existing certificate ({e!r})")
        return ["(unreadable certificate)"]

    now = datetime.datetime.now(datetime.timezone.utc)
    try:
        if cert.not_valid_after_utc < now:
            return ["(expired certificate)"]
    except AttributeError:  # older `cryptography`
        pass

    # Link-local and loopback are never how a LAN client reaches this machine,
    # so their absence is not a reason to churn the certificate.
    missing = []
    for ip in [*_local_ip_addresses(), *extra]:
        if ip.startswith(("127.", "169.254.")) or ip == "::1":
            continue
        if ip not in covered:
            missing.append(ip)
    return missing


def ensure_self_signed_cert(
    certfile: str, keyfile: str, extra_ips: tuple[str, ...] = ()
) -> tuple[str, str]:
    """Ensure a self-signed cert/key pair exists, generating one if needed.

    Returns the (certfile, keyfile) paths. The certificate's Subject
    Alternative Names include ``localhost``, this machine's hostname, and every
    detected local IP address, so accessing the server by PC name or by IP both
    work without regenerating the certificate when the IP changes.
    """
    cert_path = Path(certfile)
    key_path = Path(keyfile)

    if cert_path.exists() and key_path.exists():
        # Reusing it is only safe while it still covers where the server can be
        # reached. This module exists to survive an IP change, but reusing the
        # file unconditionally is what made it NOT survive one: the certificate
        # was issued for 10.144.3.13 and the machine later came up on
        # 10.144.3.130, so every LAN visitor got a certificate warning.
        #
        # That is not cosmetic. Chrome refuses getUserMedia on an origin with a
        # certificate error, so the microphone silently produced nothing — no
        # audio reached the VAD, no turn was ever taken, and the symptom
        # presented as "she does not speak" with a completely healthy server.
        missing = _uncovered_addresses(cert_path, extra_ips)
        if not missing:
            logger.info(f"Using existing SSL certificate: {cert_path}")
            return str(cert_path), str(key_path)
        logger.warning(
            f"SSL certificate does not cover {', '.join(missing)} — regenerating "
            "(a LAN client on that address would get a certificate warning, and "
            "the browser would then refuse microphone access)"
        )

    logger.info("Generating self-signed SSL certificate for LAN HTTPS access...")

    # Imported lazily so the dependency is only needed when HTTPS is enabled.
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    hostname = socket.gethostname()
    # extra_ips: addresses the detection below cannot see — it finds the
    # default-route interface, not e.g. an isolated cable to the PROTO.
    ip_list = sorted({*_local_ip_addresses(), *extra_ips})

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])

    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.DNSName(hostname),
    ]
    for ip in ip_list:
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            continue

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    logger.info(f"Self-signed certificate generated: {cert_path}")
    logger.info(f"Certificate covers: localhost, {hostname}, {', '.join(ip_list)}")
    return str(cert_path), str(key_path)
