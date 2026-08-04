-- Runs via --init-file on every MariaDB start. Must be idempotent.
--
-- Creates the account the benchmark connects with. We deliberately do NOT use
-- --skip-grant-tables: on MySQL 8 (and therefore AliSQL) that option implicitly
-- enables --skip-networking, which would make the server unreachable over TCP
-- from the harness container. Using a real account on both engines keeps their
-- startup and authentication paths identical.
CREATE USER IF NOT EXISTS 'bench'@'%' IDENTIFIED BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'%' WITH GRANT OPTION;
CREATE USER IF NOT EXISTS 'bench'@'localhost' IDENTIFIED BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
