-- Runs via --init-file on every AliSQL start. Must be idempotent.
--
-- mysql_native_password is requested explicitly rather than relying on the
-- 8.0 default of caching_sha2_password. Both MySQL-family engines are driven
-- through MariaDB Connector/C so that the client stack is identical and cannot
-- skew the comparison; native password is the auth plugin both servers support
-- without any negotiation differences.
CREATE USER IF NOT EXISTS 'bench'@'%' IDENTIFIED WITH mysql_native_password BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'%' WITH GRANT OPTION;
CREATE USER IF NOT EXISTS 'bench'@'localhost' IDENTIFIED WITH mysql_native_password BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'localhost' WITH GRANT OPTION;

-- Vector features ship disabled; the server flag covers startup, this covers a
-- server someone restarted without it.
SET GLOBAL vidx_disabled = OFF;
FLUSH PRIVILEGES;
