function getErrorMessage(error: unknown): string {
    if (error instanceof Error) return error.message;
    if (typeof error === "string") return error;
    try {
        return JSON.stringify(error);
    } catch {
        return String(error);
    }
}

export function isSqliteLockedError(error: unknown): boolean {
    const message = getErrorMessage(error).toLowerCase();
    return (
        message.includes("database is locked") ||
        message.includes("sqlite_busy") ||
        message.includes("(code: 5)")
    );
}

export async function withSqliteRetry<T>(
    operation: () => Promise<T>,
    options?: {
        attempts?: number;
        baseDelayMs?: number;
        maxDelayMs?: number;
    },
): Promise<T> {
    const attempts = options?.attempts ?? 6;
    const baseDelayMs = options?.baseDelayMs ?? 150;
    const maxDelayMs = options?.maxDelayMs ?? 2_000;

    let lastError: unknown;
    for (let attempt = 0; attempt < attempts; attempt++) {
        try {
            return await operation();
        } catch (error) {
            lastError = error;
            if (!isSqliteLockedError(error) || attempt === attempts - 1) {
                throw error;
            }

            const delay = Math.min(
                maxDelayMs,
                baseDelayMs * Math.pow(2, attempt),
            );
            const jitter = Math.floor(Math.random() * 75);
            await new Promise((resolve) => setTimeout(resolve, delay + jitter));
        }
    }

    throw lastError;
}

type DbLike = {
    execute: (sql: string, bindValues?: unknown[]) => Promise<unknown>;
    select: <T>(sql: string, bindValues?: unknown[]) => Promise<T>;
};

export async function executeWithSqliteRetry(
    db: DbLike,
    sql: string,
    bindValues?: unknown[],
    options?: Parameters<typeof withSqliteRetry>[1],
): Promise<void> {
    await withSqliteRetry(async () => {
        await db.execute(sql, bindValues);
    }, options);
}

export async function selectWithSqliteRetry<T>(
    db: DbLike,
    sql: string,
    bindValues?: unknown[],
    options?: Parameters<typeof withSqliteRetry>[1],
): Promise<T> {
    return withSqliteRetry(() => db.select<T>(sql, bindValues), options);
}
