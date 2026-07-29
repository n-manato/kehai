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


def ensure_self_signed_cert(certfile: str, keyfile: str) -> tuple[str, str]:
    """Ensure a self-signed cert/key pair exists, generating one if needed.

    Returns the (certfile, keyfile) paths. The certificate's Subject
    Alternative Names include ``localhost``, this machine's hostname, and every
    detected local IP address, so accessing the server by PC name or by IP both
    work without regenerating the certificate when the IP changes.
    """
    cert_path = Path(certfile)
    key_path = Path(keyfile)

    if cert_path.exists() and key_path.exists():
        logger.info(f"Using existing SSL certificate: {cert_path}")
        return str(cert_path), str(key_path)

    logger.info("Generating self-signed SSL certificate for LAN HTTPS access...")

    # Imported lazily so the dependency is only needed when HTTPS is enabled.
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    hostname = socket.gethostname()
    ip_list = _local_ip_addresses()

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, hostname)]
    )

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
