-- Runs via --init-file on every MySQL 9.7 + MyVector start, and once during
-- initialisation. Must be idempotent.
--
-- Identical in effect to docker/percona/init.sql, which explains why the
-- account names no auth plugin: MySQL 9.7 has removed mysql_native_password
-- from the server, so the default (caching_sha2_password) is the only choice.
--
-- The bench account is also the account MyVector's plugin uses to connect
-- back to this server (myvector.cnf, written by the entrypoint): the index
-- build reads the table through it and the online-update listener opens a
-- replication stream with it, which needs REPLICATION SLAVE/CLIENT. ALL
-- PRIVILEGES covers both; nothing narrower is granted because nothing
-- narrower is needed by a benchmark server on a private network.
CREATE USER IF NOT EXISTS 'bench'@'%' IDENTIFIED BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'%' WITH GRANT OPTION;
CREATE USER IF NOT EXISTS 'bench'@'localhost' IDENTIFIED BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
